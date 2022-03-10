from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator_old_mc2, ProposalTargetCreator_old_mc

from torch import nn
import torch as t
from utils import array_tool as at
#from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
import numpy as np
import cv2
from model.utils.bbox_tools import loc2bbox, bbox_iou
from model.utils.nms import non_maximum_suppression
import cupy as cp
from utils.focal_loss import focal_loss
from utils.logistic_loss import logistic_loss, logistic_loss_weight
from utils.loss_tool import IouLoss
from utils.iou_losses import iou_loss, ciou_loss, get_ious

from model.vgg16_diy import  VGG16

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cos_loss',
                        'anchor_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator_old_mc()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        #self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(2)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

        self.focalloss = focal_loss(gamma=3)
        self.logisticloss = logistic_loss()
        self.iouloss = IouLoss(losstype='Iou', gamma=2)

    def forward(self, imgs, bboxes, labels, scale, imgpath, epoch, imgnum):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features1 = self.faster_rcnn.extractor1(imgs)
        features2 = self.faster_rcnn.extractor2(features1)


        rpn_locs, rois, roi_indices, anchor, rpn_anchor_classes, rpn_coss, roi_score, old_rois = \
            self.faster_rcnn.rpn(features2, img_size, imgpath, scale, keepa=False)  # rois is 1000*4

        # save the roi in image every 20 epoch and every 50 image
        if epoch % 10 == 0 and imgnum % 5000 == 0:
            imgshow2(imgpath, bboxes[0], rois, 'RPN_anchor', epoch, imgnum, roi_score, scale, labels[0])

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        rpn_anchor_class = rpn_anchor_classes
        rpn_cos = rpn_coss[0]

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input

        sample_roi, gt_roi_loc, gt_roi_label, gt_roi_bbox, gt_roi_label2 = self.proposal_target_creator(
            roi,
            old_rois,
            at.tonumpy(bbox),
            at.tonumpy(label),
            H,
            W,
            self.loc_normalize_mean,
            self.loc_normalize_std)

        # save the sample roi in image every 20 epoch and every 50 image
        if epoch % 10 == 0 and imgnum % 5000 == 0:
            #imgshow(imgpath, bboxes[0], sample_roi, 'ROI_anchor', epoch, imgnum, roi_score, scale, label, gt_roi_label)
            imgshow(imgpath, bboxes[0], sample_roi, 'ROI_anchor', epoch, imgnum, scale)

        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score, roi_bbox = self.faster_rcnn.head(
            features1,
            features2,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        #fracture_index = np.where(at.tonumpy(label)==0)[0]
        #fracture_bbox = bbox[fracture_index, :]
        gt_rpn_loc, gt_rpn_cos, gt_rpn_label, gt_anchor_class_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size,
            scale)

        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        gt_rpn_cos = at.totensor(gt_rpn_cos)
        index_label = np.where(gt_rpn_label.cpu() > 0)[0]
        gt_rpn_cos = gt_rpn_cos[index_label]
        rpn_cos = rpn_cos[index_label]
        rpn_cos_loss = F.smooth_l1_loss(rpn_cos, gt_rpn_cos)
        # ****************the loss of class anchor get different anchor base*********************
        gt_anchor_class_label = (t.tensor(gt_anchor_class_label)).long()
        anchor_cls_loss = F.cross_entropy(rpn_anchor_class, gt_anchor_class_label.cuda(), ignore_index=-1)

        # ------------------ ROI losses (fast rcnn loss) -------------------

        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)
        gt_roi_bbox = at.totensor(gt_roi_bbox)
        keep_index = np.where(gt_roi_label.cpu() > 0)[0]

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_cls_loc.contiguous(),
            gt_roi_loc.float(),
            gt_roi_label.data,
            self.roi_sigma)

        gt_roi_label2 = at.totensor(gt_roi_label2).long()
        roi_cls_loss = self.logisticloss(roi_score, gt_roi_label2.cuda(), ignore_index=-1)

        losses = [rpn_loc_loss, rpn_cos_loss, anchor_cls_loss, roi_loc_loss, 10*roi_cls_loss]

        losses = losses + [sum(losses)]

        # compute the property
        SE, SP = compute_property(roi_score, gt_roi_label)
        return LossTuple(*losses), SE, SP

    def train_step(self, imgs, bboxes, labels, scale, imgpath, epoch, imgnum):
        self.optimizer.zero_grad()
        losses, SE, SP = self.forward(imgs, bboxes, labels, scale, imgpath, epoch, imgnum)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        #print('the losses is:')
        #print(losses)
        return losses, SE, SP

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        #save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
           # save_path = 'checkpoints/fasterrcnn_%s' % timestr
            save_path='./checkpoints/P1_%s'%timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        #self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

def imgshow2(imgpath, gtbbox, roibbox, path, epoch, imgnum, predscore, scale, label, gtlabel=[-1]):
    img=cv2.imread(imgpath)
    gtbbox=np.array(gtbbox.cpu())
    gtnum=gtbbox.shape[0]
    #print("gtbox", gtbbox)

    roibbox=np.array(roibbox)
    roinum=roibbox.shape[0]
    predscore = np.array(predscore)
    if gtlabel[0] == -1:
        if roibbox.any():
            for i in range(roinum):
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (0, 255, 0), 1)
                cv2.putText(img, str(np.round(predscore[i], 2)), (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale) - 5),
                            cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)
    else:
        for i in range(roinum):
            if gtlabel[i] == 0:
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (255, 0, 0), 1)
            elif gtlabel[i] == 1:
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (0, 255, 0), 1)
            else:
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (240, 32, 160), 1)

    for i in range(gtnum):
        if label[i] == 0:
            cv2.rectangle(img, (int(gtbbox[i, 1] / scale), int(gtbbox[i, 0] / scale)), (int(gtbbox[i, 3] / scale), int(gtbbox[i, 2] / scale)), (0, 0, 255), 1)  # x1y1,x2y2,BGR
        else:
            cv2.rectangle(img, (int(gtbbox[i, 1] / scale), int(gtbbox[i, 0] / scale)), (int(gtbbox[i, 3] / scale), int(gtbbox[i, 2] / scale)), (0, 255, 255), 1)

    #cv2.imshow('img with rectangle', img)
    #cv2.waitKey(0)
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    save_path = './imgs/'+path+'/'
    cv2.imwrite(os.path.join(save_path, 'epoch'+str(epoch)+'_'+str(imgnum) + '_' + str(filename) + '.png'), img)

def imgshow(imgpath, gtbbox, roibbox, path, epoch, imgnum, scale, gtlabel=[-1]):
    img=cv2.imread(imgpath)
    gtbbox=np.array(gtbbox.cpu())
    gtnum=gtbbox.shape[0]
    #print("gtbox", gtbbox)

    roibbox=np.array(roibbox)
    roinum=roibbox.shape[0]
    if gtlabel[0] == -1:
        if roibbox.any():
            for i in range(roinum):
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (0, 0, 255), 1)
    else:
        for i in range(roinum):
            if gtlabel[i] == 0:
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (255, 0, 0), 1)
            else:
                cv2.rectangle(img, (int(roibbox[i, 1]/scale), int(roibbox[i, 0]/scale)), (int(roibbox[i, 3]/scale), int(roibbox[i, 2]/scale)), (0, 255, 255), 1)

    for i in range(gtnum):
        cv2.rectangle(img, (int(gtbbox[i, 1] / scale), int(gtbbox[i, 0] / scale)),
                      (int(gtbbox[i, 3] / scale), int(gtbbox[i, 2] / scale)), (255, 198, 45), 2)  # x1y1,x2y2,BGR
    #cv2.imshow('img with rectangle', img)
    #cv2.waitKey(0)
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    save_path = './imgs/'+path+'/'
    cv2.imwrite(os.path.join(save_path, str(imgnum) + '_' + str(filename) + '.png'), img)