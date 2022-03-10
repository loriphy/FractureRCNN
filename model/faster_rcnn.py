from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import numpy as xp
import cupy as cp
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox, loc2bbox2, side2bbox
from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt
import cv2
import os

def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(self, extractor1, extractor2, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor1 = extractor1
        self.extractor2 = extractor2
        self.rpn = rpn
        
        #for p in self.parameters():
            #p.requires_grad = False
        
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, gtbbox, imgpath, gtlabel, scale=1., augment=0):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        h1 = self.extractor1(x)
        h2 = self.extractor2(h1)
        rpn_locs, rois, roi_indices, anchor, rpn_anchor_classes, rpn_coss, roi_score, old_rois = \
            self.rpn(h2, img_size, imgpath, scale)

        #imgshow(imgpath, gtbbox, anchor, 'TestAnchor')
        imgshow(imgpath, gtbbox, rois, gtlabel, 'TestRois', [-1], scale)

        roi_cls_locs, roi_scores, roi_bbox = self.head(
            h1, h2, rois, roi_indices)   # classification

        #--------------------------compute loss--------------------------
        # ------------------ RPN losses -------------------#
        gtbbox = gtbbox[0]
        gt_rpn_loc, gt_rpn_label = AnchorTargetCreator2(
            at.tonumpy(gtbbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs,
            gt_rpn_loc,
            gt_rpn_label.data,
            opt.roi_sigma)
        #rpn_cls_loss = F.cross_entropy(rpn_scores, gt_rpn_label.cuda(), ignore_index=-1)
        rpn_cls_loss = 0

        return roi_cls_locs, roi_scores, rois, rpn_loc_loss, rpn_cls_loss, roi_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.1
            self.score_thresh = 0.1
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        final_num = 0
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            #cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class-1, 4))[:, 0, :]
            cls_bbox_l = raw_cls_bbox.reshape(-1, 4)
            prob_l = raw_prob[:, l]
            mask = prob_l >= self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            final_num += len(keep)
        if final_num:
            bbox = np.concatenate(bbox, axis=0).astype(np.float32)
            label = np.concatenate(label, axis=0).astype(np.int32)
            score = np.concatenate(score, axis=0).astype(np.float32)
        else:
            bbox = np.array([])
            label = np.array([])
            score = np.array([])

        return bbox, label, score

    @nograd
    def predict(self, imgs, gtbbox, imgpath, gtlabel, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize: #  false
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, rpn_loc_loss, rpn_cls_loss, _ = self(img, gtbbox, imgpath, gtlabel, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            mean = 0.0
            std = 0.1
            roi_cls_loc = (roi_cls_loc * std + mean)

            roi_cls_loc = roi_cls_loc.view(-1, 1, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = side2bbox(at.tonumpy(roi).reshape((-1, 4)), at.tonumpy(roi_cls_loc).reshape((-1, 4)))

            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            if len(score) > 15:
                bbox = bbox[:15, :]
                label = label[:15]
                score = score[:15]
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores, rpn_loc_loss, rpn_cls_loss

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params, lr=opt.lrs[0])
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

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

#------------------------------------------function for loss------------------
def calc_ious( anchor, bbox, inside_index):
    # ious between the anchors and the gt boxes
    ious = bbox_iou(anchor, bbox)
    argmax_ious = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(inside_index)), argmax_ious]
    gt_argmax_ious = ious.argmax(axis=0)
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]
    return argmax_ious, max_ious, gt_argmax_ious

def create_label(inside_index, anchor, bbox):
    n_sample = 64
    pos_iou_thresh = 0.15
    neg_iou_thresh = 0
    pos_ratio = 0.5
    label = np.empty((len(inside_index),), dtype=np.int32)
    label.fill(-1)
    argmax_ious, max_ious, gt_argmax_ious = calc_ious(anchor, bbox, inside_index)
    label[max_ious <= neg_iou_thresh] = 0
    label[gt_argmax_ious] = 1
    label[max_ious >= pos_iou_thresh] = 1
    n_pos = int(pos_ratio * n_sample)
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(
            pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1

    # subsample negative labels if we have too many
    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(
            neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1

    return argmax_ious, label

def AnchorTargetCreator2(bbox,anchor,img_size):
    img_H, img_W = img_size
    n_anchor = len(anchor)
    inside_index = _get_inside_index(anchor, img_H, img_W)
    anchor = anchor[inside_index]
    argmax_ious, label = create_label(inside_index, anchor, bbox)

    # compute bounding box regression targets
    loc = bbox2loc(anchor, bbox[argmax_ious])

    # map up to original set of anchors
    label = _unmap(label, n_anchor, inside_index, fill=-1)
    loc = _unmap(loc, n_anchor, inside_index, fill=0)

    return loc, label

def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

def bbox2loc(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def imgshow(imgpath, gtbbox, predbbox, gtlabel, path, predscore, scale=1.0):
    img = cv2.imread(imgpath)
    gtbbox = np.array(gtbbox[0])
    gtclass_index = np.where(gtlabel == 0)[0]
    gtbbox = gtbbox[gtclass_index, :]
    gtnum=gtbbox.shape[0]
    predbbox=np.array(predbbox)
    prednum=predbbox.shape[0]
    predscore = np.array(predscore)
    if predbbox.any():
        for i in range(prednum):
            cv2.rectangle(img, (int(predbbox[i, 1]/scale), int(predbbox[i, 0]/scale)), (int(predbbox[i, 3]/scale), int(predbbox[i, 2]/scale)), (0, 255, 0), 1)
            if predscore[0] != -1 :
                cv2.putText(img, str(np.round(predscore[i], 2)), (int(predbbox[i, 1]/scale), int(predbbox[i, 0]/scale) - 5),
                            cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)
    for i in range(gtnum):
        cv2.rectangle(img, (gtbbox[i, 1], gtbbox[i, 0]), (gtbbox[i, 3], gtbbox[i, 2]), (0, 0, 255), 1)  # x1y1,x2y2,BGR
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    save_path = './imgs/'+path+'/'
    cv2.imwrite(os.path.join(save_path, filename + '.png'), img)
