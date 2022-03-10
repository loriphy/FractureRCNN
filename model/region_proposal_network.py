import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn
import cv2
import os

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """
    #[16, 9, 14, 21, 33, 54, 93]
    #[16, 9, 14, 23, 39, 78]
    #[16, 9, 14, 21, 33, 54, 95]
    #[16, 8, 13, 18, 27, 40, 63, 101]
    #[16, 8, 11, 16, 23, 32, 43, 64, 101]
    def __init__(
            self, in_channels=512, mid_channels=512,
            feat_stride=8, anchor_base_scales=np.array([16, 9, 14, 21, 33, 54, 93]), #16
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base_scales = anchor_base_scales
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        self.n_anchor = 1
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.loc = nn.Conv2d(mid_channels, self.n_anchor * 4, 1, 1, 0)
        self.anchor_class_scales = nn.Conv2d(mid_channels, self.anchor_base_scales.shape[0], 1, 1, 0)
        self.anchor_cos = nn.Conv2d(mid_channels, self.n_anchor, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
        normal_init(self.anchor_class_scales, 0, 0.01)
        normal_init(self.anchor_cos, 0, 0.01)
        #self.channel_attention = ChannelAttention(channel=512)
        #self.spatial_attention = SpatialAttention(kernelsize=1)

    def forward(self, x, img_size, imgpath, scale=1., keepa=True):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, channel, hh, ww = x.shape
        #'''
        attention_skeleton = _skeleton_mask(imgpath, scale, self.feat_stride, channel)
        attention_skeleton = attention_skeleton.repeat(n, 1, 1, 1).cuda()
        #'''

        #---------add a attention mask------------
        #x = self.channel_attention(x)
        #x = self.spatial_attention(x)
        h = F.relu(self.conv1(x))
        h = attention_skeleton * h
        #feature_visualization(h, imgnum)
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_cos = t.sigmoid(self.anchor_cos(h))
        rpn_cos = rpn_cos.permute(0, 2, 3, 1).contiguous().view(n, -1, 1)

        rpn_anchor_class = self.anchor_class_scales(h)
        rpn_anchor_class = rpn_anchor_class.permute(0, 2, 3, 1).contiguous()
        rpn_anchor_class_p = F.softmax(rpn_anchor_class.view(n, hh, ww, self.n_anchor, self.anchor_base_scales.shape[0]), dim=4)

        rpn_fg_scores = 1 - rpn_anchor_class_p[:, :, :, :, 0]

        rpn_anchor_class = rpn_anchor_class.view(-1, self.anchor_base_scales.shape[0])
        rpn_anchor_class_p = rpn_anchor_class_p.view(-1, self.anchor_base_scales.shape[0])
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        anchor_assignment = t.max(rpn_anchor_class_p, 1)[1]
        anchor_scale = self.anchor_base_scales[anchor_assignment.cpu()]*scale
        anchor_hw = _cos2hw(anchor_scale, rpn_cos[0].cpu().data.numpy())
        anchor_yx = _enumerate_shifted_anchor(self.feat_stride, hh, ww)
        anchor = np.zeros((hh*ww*self.n_anchor, 4), dtype=np.float32)
        anchor[:, 0] = anchor_yx[:, 0] - 0.5 * anchor_hw[:, 0]
        anchor[:, 1] = anchor_yx[:, 1] - 0.5 * anchor_hw[:, 1]
        anchor[:, 2] = anchor_yx[:, 0] + 0.5 * anchor_hw[:, 0]
        anchor[:, 3] = anchor_yx[:, 1] + 0.5 * anchor_hw[:, 1]

        is_keepa = False
        if keepa:
            keep_anchor_index = np.where(anchor_assignment.cpu().numpy() > 0)[0]
            keep_anchor = anchor[keep_anchor_index]
            keep_rpn_locs = rpn_locs[0][keep_anchor_index]
            keep_fg_scores = rpn_fg_scores[0][keep_anchor_index]
            if keep_anchor_index.shape[0] > 0:
                is_keepa = True
            else:
                is_keepa = False

        rois = list()
        old_rois = list()
        roi_indices = list()
        roi_scores = list()
        for i in range(n): # use proposal get about 2000/600 rois
            if is_keepa:
                roi, roi_score, old_roi = self.proposal_layer(
                    keep_rpn_locs.cpu().data.numpy(),
                    keep_fg_scores.cpu().data.numpy(),
                    keep_anchor, img_size,
                    scale=scale)
            else:
                roi, roi_score, old_roi = self.proposal_layer(
                    rpn_locs[i].cpu().data.numpy(),
                    rpn_fg_scores[i].cpu().data.numpy(),
                    anchor, img_size,
                    scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            old_rois.append(old_roi)
            roi_indices.append(batch_index)
            roi_scores.append(roi_score)

        rois = np.concatenate(rois, axis=0)
        old_rois = np.concatenate(old_rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        roi_scores = np.concatenate(roi_scores, axis=0)
        return rpn_locs, rois, roi_indices, anchor, rpn_anchor_class, rpn_cos, roi_scores, old_rois

def _skeleton_mask(imgpath, scale=1.0, G=8, channel=512):
    '''
        Get the skeleton of skull
    '''
    VOCPath = '../VOC110'
    #SkeletonPath = os.path.join(VOCPath, 'SkeletonImage')
    SkeletonPath = os.path.join(VOCPath, 'EdgeImage')
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    PngPath = os.path.join(SkeletonPath, filename + '.png')
    Png = cv2.imread(PngPath)
    H, W, _ = Png.shape
    Skeleton = Png.copy()
    Skeleton = cv2.resize(Skeleton, (int(H * scale), int(W * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    Skeleton = cv2.dilate(Skeleton, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    Skeleton = cv2.erode(Skeleton, kernel, iterations=1)
    H2, W2, C = Skeleton.shape
    H_new = int(H2 / G)
    W_new = int(W2 / G)
    NewPng = np.zeros((H_new, W_new, C)).astype(np.uint8)
    for h in range(H_new):
        for w in range(W_new):
            for c in range(C):
                #Skeleton[G * h:G * (h + 1), G * w:G * (w + 1), c] = np.max( Skeleton[G * h:G * (h + 1), G * w:G * (w + 1), c]).astype(np.int)
                NewPng[h, w, c] = np.max(Skeleton[G * h:G * (h + 1), G * w:G * (w + 1), c]).astype(np.uint8)
    # if dilate the skeleton
    #kernel = np.ones((16, 16), np.uint8)
    #Skeleton_dilate = cv2.dilate(NewPng, kernel)
    #b, g, r = cv2.split(Skeleton_dilate)
    b, g, r = cv2.split(NewPng)

    pngb = np.array(b)
    indexpng = np.where(pngb > 0)
    indexy = indexpng[0]
    indexx = indexpng[1]
    pngb[indexy, indexx] = 1.0
    tpng = t.from_numpy(pngb)
    tpng = tpng.float()
    tpngs = tpng.repeat(channel, 1, 1)
    return tpngs


def _enumerate_shifted_anchor(feat_stride, height, width):
    # Enumerate all shifted anchors' center:
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel()), axis=1)
    shift = shift + feat_stride/2
    return shift

def _cos2hw(ascale, acos):
    hw = np.zeros((ascale.shape[0], 2))
    # avoid the acos = 0
    index0 = np.where(acos == 0)[0]
    acos[index0] = acos[index0] + 1e-5
    ascale = ascale.reshape(-1, 1)
    hw[:, 1] = ascale[:, 0]
    #hw[:, 0] = (np.multiply(ascale, np.sqrt(1 - np.multiply(acos, acos)))/acos)[0]
    hw[:, 0] = np.multiply(ascale, np.sqrt(1/np.multiply(acos, acos)-1))[0]
    return hw

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

class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//reduction, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()
    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)
        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)
        y = self.__sigmoid(y1 + y2)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernelsize=3, masknum=2):
        super(SpatialAttention, self).__init__()
        assert kernelsize % 2 == 1, "kernel size = {}".format(kernelsize)
        padding = (kernelsize - 1) // 2
        self.__layer = nn.Sequential(
            nn.Conv2d(masknum, 1, kernel_size=kernelsize, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = t.mean(x, dim=1, keepdim=True)
        max_mask, _ = t.max(x, dim=1, keepdim=True)
        mask = t.cat([avg_mask, max_mask], dim=1)
        mask = self.__layer(mask)
        return x * mask

class FractureSpatialAttention(nn.Module):

    def __init__(self, kernelsize=1, masknum=3):
        super(FractureSpatialAttention, self).__init__()
        assert kernelsize % 2 == 1, "kernel size = {}".format(kernelsize)
        padding = (kernelsize - 1) // 2
        self.__layer = nn.Sequential(
            nn.Conv2d(masknum, 1, kernel_size=kernelsize, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x, imgpath, scale, feat_stride, channel):
        avg_mask = t.mean(x, dim=1, keepdim=True)
        max_mask, _ = t.max(x, dim=1, keepdim=True)
        skeleton = t.ones_like(max_mask)
        skeleton[:, 0, :, :] = _skeleton_mask(imgpath, scale, feat_stride, max_mask.shape[0])
        mask = t.cat([avg_mask, max_mask, skeleton], dim=1)
        mask = self.__layer(mask)
        return x * mask

def feature_visualization(features, imgnum):
    feature = features[0, :, :, :].cpu().detach().numpy()
    img = np.sum(feature, axis=0)
    img = (img - np.min(img))/(np.max(img) - np.min(img)) * 512
    img = img.astype(np.uint8)
    save_path = './imgs/' + 'feature' + '/'
    cv2.imwrite(os.path.join(save_path, '_' + str(imgnum) + '.png'), img)