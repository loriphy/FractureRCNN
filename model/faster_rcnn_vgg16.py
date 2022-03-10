from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16_bn
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt
from torchvision.ops import roi_align, roi_pool, RoIPool, RoIAlign
import torch.nn.functional as F
from utils.FcaNet import get_2d_dct, keep_k_frequency

class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 8  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=1,#can be change according to your classes
                 #ratios=[0.5, 1, 2],
                 ratios=[0.6, 1.2, 2],#y/x,0.75,1,1.25
                 #anchor_scales=[8, 16, 32]
                 anchor_scales=[1.4, 3, 6.8]
                 ):
                 
        #extractor, classifier = decom_vgg16()
        #'''
        extractor1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),         # 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),        # 5   + 2
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # 6   + 1

            nn.Conv2d(64, 128, kernel_size=3, padding=1),       # 10  + 2*2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),      # 14  + 2*2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # 16  + 1*2
        )

        extractor2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),      # 24  + 2*4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),      # 32  + 2*4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),      # 40  + 2*4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # 44  + 1*4

            nn.Conv2d(256, 512, kernel_size=3, padding=1),      # 60  + 2*8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),      # 76  + 2*8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),      # 92  + 2*8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        rpn = RegionProposalNetwork(
            512, 512,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead_Side2(     # Side2
            n_class=n_fg_class + 1,
            roi_size=5,
            spatial_scale=(1. / self.feat_stride),
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor1,
            extractor2,
            rpn,
            head,
        )

class VGG16RoIHead_Side2(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    Head with channel attention and spatial attention
    """

    def __init__(self, n_class, roi_size, spatial_scale):
        # n_class includes the background
        super(VGG16RoIHead_Side2, self).__init__()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.side_size1 = 3
        self.side_size2 = 5

        self.classifier1 = nn.Sequential(
            nn.Linear(512 * self.roi_size * self.roi_size, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(2048*4, 1024),
            nn.ReLU(True)
        )

        self.side_encoder = nn.Sequential(
            nn.Linear(512 * self.side_size1 * self.side_size2, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True)
        )

        self.loc_left = nn.Linear(2048, 1)
        self.loc_right = nn.Linear(2048, 1)
        self.loc_top = nn.Linear(2048, 1)
        self.loc_bottom = nn.Linear(2048, 1)
        self.score = nn.Linear(4096+1024, n_class)

        normal_init(self.loc_left, 0, 0.01)
        normal_init(self.loc_right, 0, 0.01)
        normal_init(self.loc_top, 0, 0.01)
        normal_init(self.loc_bottom, 0, 0.01)
        normal_init(self.score, 0, 0.01)

    def forward(self, x0, x, rois, roi_indices, add=True):
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy, (B, x1, y1, x2, y2)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        # get the useful information area for the for sides of the object box (b, x1, y1, x2, y2)
        sw1 = 2 * t.log2(indices_and_rois[:, 3] - indices_and_rois[:, 1])
        sw2 = 0.2 * (indices_and_rois[:, 3] - indices_and_rois[:, 1])
        side_width = t.max(t.cat((sw1.reshape(-1, 1), sw2.reshape(-1, 1)), 1), 1)[0]
        sh1 = 2 * t.log2(indices_and_rois[:, 4] - indices_and_rois[:, 2])
        sh2 = 0.2 * (indices_and_rois[:, 4] - indices_and_rois[:, 2])
        side_height = t.max(t.cat((sh1.reshape(-1, 1), sh2.reshape(-1, 1)), 1), 1)[0]
        # for the left: xl1 = x1-0.5*1/sita*w, yl1 = y1, xl2 = x1+0.5*1/sita*w, yl2 = y2
        areas_left = indices_and_rois * 1.0
        areas_left[:, 1] = areas_left[:, 1] - 0.5 * side_width
        areas_left[:, 3] = areas_left[:, 1] + 0.5 * side_width
        areas_left[:, 1:5] = t.clamp(areas_left[:, 1:5], 0, int(x.shape[-1] / self.spatial_scale))
        # for the right: xr1 = x2-0.5*1/sita*w, yr1 = y1, xr2 = x2+0.5*1/sita*w, yr2 = y2
        areas_right = indices_and_rois * 1.0
        areas_right[:, 1] = areas_right[:, 3] - 0.5 * side_width
        areas_right[:, 3] = areas_right[:, 3] + 0.5 * side_width
        areas_right[:, 1:5] = t.clamp(areas_right[:, 1:5], 0, int(x.shape[-1] / self.spatial_scale))
        # for the top: xt1 = x1, yt1 = y1-0.5*1/sita*h, xt2 = x2, yt2 = y1+0.5*1/sita*h
        areas_top = indices_and_rois * 1.0
        areas_top[:, 2] = areas_top[:, 2] - 0.5 * side_height
        areas_top[:, 4] = areas_top[:, 2] + 0.5 * side_height
        areas_top[:, 1:5] = t.clamp(areas_top[:, 1:5], 0, int(x.shape[-1] / self.spatial_scale))
        # for the bottom: xt1 = x1, yt1 = y2-0.5*1/sita*h, xt2 = x2, yt2 = y2+0.5*1/sita*h
        areas_bottom = indices_and_rois * 1.0
        areas_bottom[:, 2] = areas_bottom[:, 4] - 0.5 * side_height
        areas_bottom[:, 4] = areas_bottom[:, 4] + 0.5 * side_height
        areas_bottom[:, 1:5] = t.clamp(areas_bottom[:, 1:5], 0, int(x.shape[-1] / self.spatial_scale))
        # roi_align to get the feature
        pool_left = roi_align(x, areas_left, (self.side_size1, self.side_size2), self.spatial_scale)
        pool_right = roi_align(x, areas_right, (self.side_size1, self.side_size2), self.spatial_scale)
        pool_top = roi_align(x, areas_top, (self.side_size2, self.side_size1), self.spatial_scale)
        pool_bottom = roi_align(x, areas_bottom, (self.side_size2, self.side_size1), self.spatial_scale)

        # flatten the pool
        pool_left = pool_left.view(pool_left.size(0), -1)
        pool_right = pool_right.view(pool_right.size(0), -1)
        pool_top = pool_top.view(pool_top.size(0), -1)
        pool_bottom = pool_bottom.view(pool_bottom.size(0), -1)

        # predict the offset of each side
        fc_left = self.side_encoder(pool_left)
        fc_right = self.side_encoder(pool_right)
        fc_top = self.side_encoder(pool_top)
        fc_bottom = self.side_encoder(pool_bottom)
        diff_left = self.loc_left(fc_left)
        diff_right = self.loc_right(fc_right)
        diff_top = self.loc_top(fc_top)
        diff_bottom = self.loc_bottom(fc_bottom)

        # revise the predicted box
        # '''
        mean = 0.0
        std = 0.1
        diff_left_bn = diff_left * std + mean
        diff_top_bn = diff_top * std + mean
        diff_right_bn = diff_right * std + mean
        diff_bottom_bn = diff_bottom * std + mean
        # '''

        fine_rois = indices_and_rois * 1.0
        fine_rois[:, 1] = diff_left_bn.reshape(-1) * (fine_rois[:, 3] - fine_rois[:, 1]) + fine_rois[:, 1]
        fine_rois[:, 2] = diff_top_bn.reshape(-1) * (fine_rois[:, 4] - fine_rois[:, 2]) + fine_rois[:, 2]
        fine_rois[:, 3] = diff_right_bn.reshape(-1) * (fine_rois[:, 3] - fine_rois[:, 1]) + fine_rois[:, 3]
        fine_rois[:, 4] = diff_bottom_bn.reshape(-1) * (fine_rois[:, 4] - fine_rois[:, 2]) + fine_rois[:, 4]

        # predict the score of the boxes based on the fine box
        pool_center = roi_align(x, fine_rois, (self.roi_size, self.roi_size), self.spatial_scale)
        pool_center = pool_center.view(pool_center.size(0), -1)
        f_center = self.classifier1(pool_center)
        pool_boundary = t.cat([fc_left, fc_right, fc_top, fc_bottom], dim=1)
        f_boundary = self.classifier2(pool_boundary)
        roi_scores = self.score(t.cat([f_center, f_boundary], dim=1))

        return t.cat((diff_left, diff_right, diff_top, diff_bottom), 1), roi_scores, fine_rois[:, [2, 1, 4, 3]]

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()