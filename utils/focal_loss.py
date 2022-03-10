from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

class focal_loss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            #assert alpha < 1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels, ignore_index=-1):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        keep_index = np.where(labels.cpu() != ignore_index)[0]
        preds = preds[keep_index, :]
        labels = labels[keep_index]
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))   #  the same as nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        a = self.alpha
        # only keep the hard sample
        '''
        posnum = (np.where(labels.cpu() > 0)[0]).shape[0]
        keep1 = np.where(preds_softmax[:posnum].cpu() < 0.99)[0]
        keep2 = np.where(preds_softmax[posnum:].cpu() < 0.8)[0] + posnum
        hardsemple = np.append(keep1, keep2)
        if hardsemple.shape[0] > 0:
            preds_softmax = preds_softmax[hardsemple]
            preds_logsoft = preds_logsoft[hardsemple]
            a = a[hardsemple]
        '''

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) =  (1-pt)**γ in focal loss
        loss = torch.mul(a, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

'''
focalloss = focal_loss()
a1 = torch.tensor([[-0.0267, -0.0138], [0.0093, -0.0010], [0.0036,-0.0367]])
gt = torch.tensor([-1, 1, 0])
loss = focalloss(a1, gt)
'''

#rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
