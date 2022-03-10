import numpy as np
import torch
import math

def get_iou(pred_bbox, gt_bbox):
    '''
    :param pred_bbox: [y1, x1, y2, x2]
    :param gt_bbox:  [y1, x1, y2, x2]
    :return: iou
    '''

    iymin = max(pred_bbox[0], gt_bbox[0])
    ixmin = max(pred_bbox[1], gt_bbox[1])
    iymax = min(pred_bbox[2], gt_bbox[2])
    ixmax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.)
    ih = np.maximum(iymax - iymin + 1.0, 0.)

    inters = iw * ih

    # uni=s1+s2-inters
    uni = (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0) + \
          (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0) - inters

    iou = inters / uni

    return iou

def get_ious(preds, bbox):
    '''
    :param preds:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param bbox:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :return: ious
    '''

    y1 = torch.max(preds[:, 0], bbox[:, 0])
    x1 = torch.max(preds[:, 1], bbox[:, 1])
    y2 = torch.min(preds[:, 2], bbox[:, 2])
    x2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)

    inters = w * h

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    ious = inters / uni

    return ious

def iou_loss(preds, bbox, eps=1e-6, reduction='mean'):
    '''
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
    :param preds:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param bbox:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :return: loss
    '''
    y1 = torch.max(preds[:, 0], bbox[:, 0])
    x1 = torch.max(preds[:, 1], bbox[:, 1])
    y2 = torch.min(preds[:, 2], bbox[:, 2])
    x2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)

    inters = w * h

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    ious = (inters / uni).clamp(min=eps)
    #loss = -ious.log()
    loss = 1 - ious

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss

def giou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
   https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    :param preds:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param bbox:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :return: loss
    '''
    iy1 = torch.max(preds[:, 0], bbox[:, 0])
    ix1 = torch.max(preds[:, 1], bbox[:, 1])
    iy2 = torch.min(preds[:, 2], bbox[:, 2])
    ix2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)

    # overlap
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps

    # ious
    ious = inters / uni

    ey1 = torch.min(preds[:, 0], bbox[:, 0])
    ex1 = torch.min(preds[:, 1], bbox[:, 1])
    ey2 = torch.max(preds[:, 2], bbox[:, 2])
    ex2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)

    # enclose erea
    enclose = ew * eh + eps

    giou = ious - (enclose - uni) / enclose

    loss = 1 - giou

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss

def diou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param bbox:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    iy1 = torch.max(preds[:, 0], bbox[:, 0])
    ix1 = torch.max(preds[:, 1], bbox[:, 1])
    iy2 = torch.min(preds[:, 2], bbox[:, 2])
    ix2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cypreds = (preds[:, 2] + preds[:, 0]) / 2
    cxpreds = (preds[:, 3] + preds[:, 1]) / 2

    cybbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cxbbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    oy1 = torch.min(preds[:, 0], bbox[:, 0])
    ox1 = torch.min(preds[:, 1], bbox[:, 1])
    oy2 = torch.max(preds[:, 2], bbox[:, 2])
    ox2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)

    diou_loss = 1 - diou

    if reduction == 'mean':
        loss = torch.mean(diou_loss)
    elif reduction == 'sum':
        loss = torch.sum(diou_loss)
    else:
        raise NotImplementedError
    return loss

def ciou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param bbox:[[y1,x1,y2,x2], [y1,x1,y2,x2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    iy1 = torch.max(preds[:, 0], bbox[:, 0])
    ix1 = torch.max(preds[:, 1], bbox[:, 1])
    iy2 = torch.min(preds[:, 2], bbox[:, 2])
    ix2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cypreds = (preds[:, 2] + preds[:, 0]) / 2
    cxpreds = (preds[:, 3] + preds[:, 1]) / 2

    cybbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cxbbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    oy1 = torch.min(preds[:, 0], bbox[:, 0])
    ox1 = torch.min(preds[:, 1], bbox[:, 1])
    oy2 = torch.max(preds[:, 2], bbox[:, 2])
    ox2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag

    # calculate v,alpha
    hbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    wbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    hpreds = preds[:, 2] - preds[:, 0] + 1.0
    wpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ciou_loss = 1 - ciou
    if reduction == 'mean':
        loss = torch.mean(ciou_loss)
    elif reduction == 'sum':
        loss = torch.sum(ciou_loss)
    else:
        raise NotImplementedError
    return loss
