import numpy as np
import cupy as cp

from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox, bbox2side
from model.utils.nms import non_maximum_suppression
import math
import random

class ProposalTargetCreator_old(object):
    def __init__(self,
                 n_sample=64,  # 64
                 pos_ratio=0.25, pos_iou_thresh=0.2,  # can be change
                 neg_iou_thresh_hi=0.001, neg_iou_thresh_lo=0.0,
                 iou_threshs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                 ):
        self.n_sample = n_sample
        self.addposnum = 10
        self.pos_ratio = pos_ratio
        self.iou_threshs = np.array(iou_threshs)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, old_roi, bbox, label, H, W,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        n_bbox, _ = bbox.shape
        roi = np.concatenate((bbox, roi), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)

        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        #pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_index, tiou = self._assign_positive_sample(max_iou)

        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        #neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_index = self._assign_negative_sample(max_iou, tiou, roi, bbox)

        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2side(sample_roi, bbox[gt_assignment[keep_index]])
        mean = 0.0
        std = 0.1
        gt_roi_loc = (gt_roi_loc - mean) / std
        #gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        #gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label, bbox[gt_assignment[keep_index]]

    def _assign_positive_sample(self, maxiou):
        '''
        function: gain the positive sample index, set iou_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] \
                depend on how many iou there have in total.
        '''
        posnum = self.pos_ratio * self.n_sample
        #                                               tiou = self.iou_threshs[0]
        # here use dichotomy, [left, right), the answer is left
        left = 0
        right = self.iou_threshs.shape[0]
        while right - left > 1:
            mid = int((left + right)/2)
            mid_pos_index = np.where(maxiou >= self.iou_threshs[mid])[0]
            if mid_pos_index.size > posnum:
                left = mid
            else:
                right = mid
        return np.where(maxiou >= self.iou_threshs[left])[0], self.iou_threshs[left]

    def _assign_negative_sample(self, maxiou, tiou, roi, bbox, nratio=1/3, cfpratio=1/2):
        '''
        function: gain the negative sample index
            the negative sample index should contain 2 parts:1)closed false positive 2)similar false positive
        '''
        cfp_index = np.where((maxiou > 0) & (maxiou < tiou * nratio))[0]
        neg_sample = self.n_sample - self.n_sample*self.pos_ratio
        cfp_sample = int(neg_sample*cfpratio)

        distance = self._bbox_distance(roi, bbox)
        argmin_distance = distance.argmin(axis=1)
        min_distance = distance.min(axis=1)
        min_bbox = bbox[argmin_distance]
        min_bsize = min_bbox[:, 2:4] - min_bbox[:, 0:2]
        min_bsize = np.sqrt(min_bsize[:, 0] ** 2 + min_bsize[:, 1] ** 2)

        if cfp_index.size > cfp_sample:
            cfp_index = np.random.choice(cfp_index, size=cfp_sample, replace=False)
        else:
            # here we need to choose some closed false sample with iou = 0
            # the bboxes have 0 iou with ground truth, but it's center is closed to ground truth
            closed_index = np.where((min_distance < 1.5 * min_bsize) & (maxiou <= 0))[0]
            restcfpneg = int(cfp_sample-cfp_index.size)
            restcfpneg = int(min(restcfpneg, closed_index.size))
            closed_index = np.random.choice(closed_index, size=restcfpneg, replace=False)
            cfp_index = np.concatenate((cfp_index, closed_index), axis=0)

        similar_num = int(max(neg_sample-cfp_sample, neg_sample-cfp_index.size))
        similar_index = np.where(maxiou <= 0)[0]
        similar_num = int(min(similar_num, similar_index.size))
        similar_index = np.random.choice(similar_index, size=similar_num, replace=False)
        neg_index = np.concatenate((cfp_index, similar_index), axis=0)
        return neg_index

    def _bbox_distance(self, roi, bbox):
        '''
        function: caculate the the center distance between roi and bbox
        args:
            roi(array): the region of interest, shape is (N, 4)
            bbox(array): the ground truth bbox, shape is (K, 4)
        returns:
            distance(array): shape is (N, K)
        '''
        if roi.shape[1] != 4 or bbox.shape[1] != 4:
            raise IndexError
        roi_center = (roi[:, 2:4] + roi[:, 0:2])/2.
        bbox_center = (bbox[:, 2:4] + bbox[:, 0:2])/2.
        dif_wh = roi_center[:, None, :] - bbox_center[:, :]
        distance = np.sqrt(dif_wh[:, :, 0]**2 + dif_wh[:, :, 1]**2)
        return distance

class ProposalTargetCreator_old_mc(object):
    def __init__(self,
                 n_sample=64,  # 64
                 pos_ratio=0.25, pos_iou_thresh=0.2,  # can be change
                 neg_iou_thresh_hi=0.001, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, old_roi, bbox, label, H, W,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        n_bbox, _ = bbox.shape
        roi = np.concatenate((bbox, roi), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        gt_roi_label = label[gt_assignment] + 1
        max_iou = iou.max(axis=1)
        fracture_index = list()
        disturb_index = list()
        for i in range(bbox.shape[0]):
            iou_bi = iou[:, i]
            iou_bi_descend = np.sort(-iou_bi) * -1
            iou_bi_descend_index = np.argsort(-iou_bi)
            gt_bi_descend = gt_roi_label[iou_bi_descend_index]
            if label[i] == 0:
                # select sample for the fracture ground truth
                iou_bi_over_thresh_index = np.where((iou_bi_descend >= self.pos_iou_thresh) & (gt_bi_descend == 1))[0]
                if iou_bi_over_thresh_index.shape[0] > 10:
                    iou_bi_over_thresh_index = iou_bi_over_thresh_index[:10]
                fracture_index.append(iou_bi_descend_index[iou_bi_over_thresh_index])
            if label[i] == 1:
                # select sample for the disturb ground truth
                iou_bi_over_thresh_index = np.where((iou_bi_descend >= self.pos_iou_thresh) & (gt_bi_descend == 2))[0]
                if iou_bi_over_thresh_index.shape[0] > 5:
                    iou_bi_over_thresh_index = iou_bi_over_thresh_index[:5]
                disturb_index.append(iou_bi_descend_index[iou_bi_over_thresh_index])
        fracture_index = np.concatenate(np.array(fracture_index), axis=0)
        if np.where(label == 1)[0].shape[0] > 0:
            disturb_index = np.concatenate(np.array(disturb_index), axis=0)
        else:
            disturb_index = np.array([]).astype(np.int)

        fracture_per_this_image = int(min(pos_roi_per_image, fracture_index.size))
        if fracture_index.size > 0:
            fracture_index = np.random.choice(
                fracture_index, size=fracture_per_this_image, replace=False)
        disturb_per_this_image = int(min(2*pos_roi_per_image, disturb_index.size))
        if disturb_index.size > 0:
            disturb_index = np.random.choice(
                disturb_index, size=disturb_per_this_image, replace=False)

        # Select background RoIs as those within
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - fracture_per_this_image - disturb_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(fracture_index, disturb_index)
        keep_index = np.append(keep_index, neg_index)
        gt_roi_label1 = gt_roi_label[keep_index]
        gt_roi_label1[fracture_per_this_image + disturb_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]
        gt_roi_label2 = gt_roi_label[keep_index]
        gt_roi_label2[fracture_per_this_image:] = 0

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2side(sample_roi, bbox[gt_assignment[keep_index]])
        mean = 0.0
        std = 0.1
        gt_roi_loc = (gt_roi_loc - mean) / std
        return sample_roi, gt_roi_loc, gt_roi_label1, bbox[gt_assignment[keep_index]], gt_roi_label2

class ProposalTargetCreator_old_mc2(object):
    def __init__(self,
                 n_sample=80,  # 64
                 pos_ratio=0.5, pos_iou_thresh=0.2,  # can be change
                 neg_iou_thresh_hi=0.001, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, old_roi, bbox, label, H, W,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        n_bbox, _ = bbox.shape
        roi = np.concatenate((bbox, roi), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        gt_roi_label = label[gt_assignment] + 1
        max_iou = iou.max(axis=1)
        fracture_index = list()
        disturb_index = list()
        for i in range(bbox.shape[0]):
            iou_bi = iou[:, i]
            iou_bi_descend = np.sort(-iou_bi) * -1
            iou_bi_descend_index = np.argsort(-iou_bi)
            gt_bi_descend = gt_roi_label[iou_bi_descend_index]
            if label[i] == 0:
                # select sample for the fracture ground truth
                iou_bi_over_thresh_index = np.where((iou_bi_descend >= 0.5) & (gt_bi_descend == 1))[0]
                if iou_bi_over_thresh_index.shape[0] > 15:
                    iou_bi_over_thresh_index = iou_bi_over_thresh_index[:15]
                else:
                    if iou_bi_over_thresh_index.shape[0] < 10:
                        iou_bi_over_thresh_index2 = np.where((iou_bi_descend >= self.pos_iou_thresh) & (gt_bi_descend == 1))[0]
                        iou_bi_over_thresh_index = iou_bi_over_thresh_index2[: iou_bi_over_thresh_index.shape[0] + 5]
                fracture_index.append(iou_bi_descend_index[iou_bi_over_thresh_index])

            if label[i] == 1:
                # select sample for the disturb ground truth
                iou_bi_over_thresh_index = np.where((iou_bi_descend >= self.pos_iou_thresh+0.1) & (gt_bi_descend == 2))[0]
                if iou_bi_over_thresh_index.shape[0] > 5:
                    iou_bi_over_thresh_index = iou_bi_over_thresh_index[:5]
                disturb_index.append(iou_bi_descend_index[iou_bi_over_thresh_index])

        fracture_index = np.concatenate(np.array(fracture_index), axis=0)
        if np.where(label == 1)[0].shape[0] > 0:
            disturb_index = np.concatenate(np.array(disturb_index), axis=0)
        else:
            disturb_index = np.array([]).astype(np.int)

        fracture_per_this_image = int(min(pos_roi_per_image, fracture_index.size))
        if fracture_index.size > 0:
            fracture_index = np.random.choice(
                fracture_index, size=fracture_per_this_image, replace=False)
        disturb_per_this_image = int(min(pos_roi_per_image, disturb_index.size))
        if disturb_index.size > 0:
            disturb_index = np.random.choice(
                disturb_index, size=disturb_per_this_image, replace=False)

        # Select background RoIs as those within
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - fracture_per_this_image - disturb_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(fracture_index, disturb_index)
        keep_index = np.append(keep_index, neg_index)
        gt_roi_label1 = gt_roi_label[keep_index]
        gt_roi_label1[fracture_per_this_image + disturb_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]
        gt_roi_label2 = gt_roi_label[keep_index]
        gt_roi_label2[fracture_per_this_image:] = 0

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2side(sample_roi, bbox[gt_assignment[keep_index]])
        mean = 0.0
        std = 0.1
        gt_roi_loc = (gt_roi_loc - mean) / std
        return sample_roi, gt_roi_loc, gt_roi_label1, bbox[gt_assignment[keep_index]], gt_roi_label2

class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=64,
                 pos_iou_thresh=0.15, neg_iou_thresh=0,
                 pos_ratio=0.5,
                 iou_threshs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
        self.iou_threshs = np.array(iou_threshs)
        self.anchor_scales = np.array([16, 9, 14, 21, 33, 54, 93])#16

    def __call__(self, bbox, anchor, img_size, scale):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label, anchor_class_label = self._create_label(
            inside_index, anchor, bbox, self.anchor_scales * scale)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])
        cos = _bbox2cos(bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        cos = _unmap(cos, n_anchor, inside_index, fill=1)
        anchor_class_label = _unmap(anchor_class_label, n_anchor, inside_index, fill=-1)

        return loc, cos, label, anchor_class_label

    def _create_label(self, inside_index, anchor, bbox, anchor_scales):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        # anchor_class_label: i>=0 means the class of the anchor , -1 is dont care
        anchor_class_label = np.empty((len(inside_index),), dtype=np.int32)
        anchor_class_label.fill(-1)

        # use min center position inplace the max iou
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious <= self.neg_iou_thresh] = 0
        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1
        # positive label: above threshold IOU
        # max_pos_iou_thresh = self._get_pos_iou_thresh(max_ious)
        # label[max_ious >= max_pos_iou_thresh] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]

        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        # assign the anchor class label
        pos_index = np.where(label == 1)[0]
        # print("pos", pos_index.shape)
        bbox_label = np.zeros((bbox.shape[0],))
        bbox_anchor_differ = _bbox_anchor_differ(anchor_scales[1:], bbox)
        argmin_differ = bbox_anchor_differ.argmin(axis=1)
        bbox_label = bbox_label + argmin_differ + 1
        anchor_class_label[pos_index] = bbox_label[argmax_ious[pos_index]]
        neg_index = np.where(label == 0)[0]
        anchor_class_label[neg_index] = 0
        return argmax_ious, label, anchor_class_label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        return argmax_ious, max_ious, gt_argmax_ious

    def _get_pos_iou_thresh(self, maxiou):
        '''
        function: gain the positive sample index, set iou_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] \
                depend on how many iou there have in total.
        '''
        posnum = self.pos_ratio * self.n_sample
        #                                               tiou = self.iou_threshs[0]
        # here use dichotomy, [left, right), the answer is left
        left = 0
        right = self.iou_threshs.shape[0]
        while right - left > 1:
            mid = int((left + right) / 2)
            mid_pos_index = np.where(maxiou >= self.iou_threshs[mid])[0]
            if mid_pos_index.size > posnum:
                left = mid
            else:
                right = mid
        return self.iou_threshs[left]


def _bbox2cos(bbox):
    bbox_hw = bbox[:, 2:4] - bbox[:, 0:2]
    bcos = bbox_hw[:, 1] / (np.sqrt(bbox_hw[:, 1] ** 2 + bbox_hw[:, 0] ** 2))
    return bcos.reshape(-1, 1)


def _bbox_anchor_differ(ascale, bbox_b):
    wa = ascale
    wb = bbox_b[:, 3] - bbox_b[:, 1]
    differ = np.abs(wb[:, None] - wa)
    return differ


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


class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=1500,  # 1500
                 n_train_post_nms=300,  # maybe less than 2000 #300
                 n_test_pre_nms=800,  # 1000
                 n_test_post_nms=100,  # 200
                 min_size=2,
                 score_thresh=0.0001
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
        self.score_thresh = score_thresh

    def __call__(self, loc, score,
                 anchor, img_size, scale=1., keep_ratio=1):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]

        # the target is small enouph
        keep0 = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep0, :]
        score = score[keep0]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        order = score.ravel().argsort()[::-1]

        '''
        scores = score[order]
        keeps = np.where(scores > scores[100])[0]
        num = keeps.shape[0]
        f = "num.txt"
        with open(f, "a") as file:
            file.write(str(num) + '\n')
        '''

        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep0 = np.where(score >= self.score_thresh)[0]
        if keep0.shape[0] > 0:
            roi = roi[keep0, :]
            score = score[keep0]

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep1 = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)), self.nms_thresh, score)
        keep2 = non_maximum_suppression(cp.ascontiguousarray(cp.asarray(roi)), 0.7, score)

        if n_post_nms > 0:
            keep1 = keep1[:n_post_nms]
            keep2 = keep2[:n_post_nms]

        roi1 = roi[keep1]
        score1 = score[keep1]
        roi2 = roi[keep2]
        return roi1, score1, roi2
