import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
    this python file is used for compute every kinds of loss, and it contained:
    1. Iouloss: include iouloss, giouloss, diouloss, ciouloss
    2. focalloss
'''

# 1 the every kinds of iou loss ----------------------------------------------------------------------------------------
class IouLoss(nn.Module):
    def __init__(self, losstype='Giou', focal=False, gamma=2, size_average=True, pred_mode='None', variances=None):
        super(IouLoss, self).__init__()
        self.losstype = losstype
        self.size_average = size_average
        self.pred_mode = pred_mode
        self.variances = variances
        self.focal = focal
        self.gamma = gamma

    def forward(self, predloc, gtloc, priordata=-1):
        num = predloc.shape[0]

        if self.pred_mode == 'Center':
            pred_bbox = self._decode(predloc, priordata, self.variances)
        else:
            pred_bbox = predloc

        if self.losstype == 'Iou':
            loss = 1.0 - self._bbox_overlaps_iou(pred_bbox, gtloc)
        else:
            if self.losstype == 'Giou':
                loss = 1.0 - self._bbox_overlaps_giou(pred_bbox, gtloc)
            else:
                if self.losstype == 'Diou':
                    loss = 1.0 - self._bbox_overlaps_diou(pred_bbox, gtloc)
                else:
                    loss = 1.0 - self._bbox_overlaps_ciou(pred_bbox, gtloc)

        if self.focal:
            loss = torch.mul(torch.pow(loss, self.gamma), loss)

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


    def _decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
            the encoding we did for offset regression at train time.
            Args:
                loc (tensor): location predictions for loc layers,
                    Shape: [num_priors,4]
                priors (tensor): Prior boxes in center-offset form.
                    Shape: [num_priors,4].
                variances: (list[float]) Variances of priorboxes
            Return:
                decoded bounding box predictions
            """
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _bbox_overlaps_iou(self, bboxes1, bboxes2):
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        ious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return ious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            ious = torch.zeros((cols, rows))
            exchange = True
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]
        union = area1 + area2 - inter_area
        ious = inter_area / union
        ious = torch.clamp(ious, min=0, max=1.0)
        if exchange:
            ious = ious.T
        return ious

    def _bbox_overlaps_giou(self, bboxes1, bboxes2):
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        ious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return ious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            ious = torch.zeros((cols, rows))
            exchange = True
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_area = outer[:, 0] * outer[:, 1]
        union = area1 + area2 - inter_area
        closure = outer_area

        ious = inter_area / union - (closure - union) / closure
        ious = torch.clamp(ious, min=-1.0, max=1.0)
        if exchange:
            ious = ious.T
        return ious

    def _bbox_overlaps_diou(self, bboxes1, bboxes2):
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        dious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return dious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            dious = torch.zeros((cols, rows))
            exchange = True

        w1 = bboxes1[:, 2] - bboxes1[:, 0]
        h1 = bboxes1[:, 3] - bboxes1[:, 1]
        w2 = bboxes2[:, 2] - bboxes2[:, 0]
        h2 = bboxes2[:, 3] - bboxes2[:, 1]

        area1 = w1 * h1
        area2 = w2 * h2
        center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
        center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
        center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
        center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

        inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
        union = area1 + area2 - inter_area
        dious = inter_area / union - (inter_diag) / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)
        if exchange:
            dious = dious.T
        return dious

    def _bbox_overlaps_ciou(self, bboxes1, bboxes2):
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        cious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return cious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            cious = torch.zeros((cols, rows))
            exchange = True

        w1 = bboxes1[:, 2] - bboxes1[:, 0]
        h1 = bboxes1[:, 3] - bboxes1[:, 1]
        w2 = bboxes2[:, 2] - bboxes2[:, 0]
        h2 = bboxes2[:, 3] - bboxes2[:, 1]

        area1 = w1 * h1
        area2 = w2 * h2

        center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
        center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
        center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
        center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

        inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
        union = area1 + area2 - inter_area
        u = (inter_diag) / outer_diag
        iou = inter_area / union
        with torch.no_grad():
            arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
            S = 1 - iou
            alpha = v / (S + v)
            w_temp = 2 * w1
        ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
        cious = iou - (u + alpha * ar)
        cious = torch.clamp(cious, min=-1.0, max=1.0)
        if exchange:
            cious = cious.T
        return cious



