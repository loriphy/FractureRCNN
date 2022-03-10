from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

class logistic_loss(nn.Module):

    def __init__(self, alpha=0.3, num_classes=2, factor=[1.0, 1.0, 0.8], size_average=True):
        """
        logistic loss function
        :param num_classes:     the number of classes
        :param size_average:    the way of compute loss, default by mean value
        """
        super(logistic_loss, self).__init__()
        self.size_average = size_average
        self.factor = torch.tensor(factor)
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print("logistic_loss alpha = {}, please deassign it".format(alpha))
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            #self.alpha[1:] += (1-alpha)
            self.alpha[1:] += alpha

    def forward(self, preds, labels, ignore_index=-1, sort_loss=True):
        """
        logistic_loss compute loss
        :param preds: predict class. size:[B,N,C] or [B,C]    B is the batch size, N is the number of predict boxes, C is the number of class
        :param labels:  the ground truth's class label. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        keep_index = np.where(labels.cpu() != ignore_index)[0]
        preds = preds[keep_index, :]
        labels = labels[keep_index]
        preds_softmax = F.softmax(preds, dim=1)
        #preds_softmax[:, 0] = preds_softmax[:, 0] + preds_softmax[:, 2]
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))   #  the same as nll_loss ( crossempty = log_softmax + nll )

        R = 10.9861
        P0 = 0.5
        K = 1.0
        beta = 1.5
        self.alpha = self.alpha.to(preds.device)
        self.factor = self.factor.to(preds.device)
        a = self.alpha.gather(0, labels.view(-1)).reshape(-1, 1)
        factor = self.factor.gather(0, labels.view(-1))
        h = K*P0*torch.exp(R*(-(1-a)))/(K+P0*(torch.exp(R*(-(1-a)))-1))
        loss = (K*P0*torch.exp(R*(-(preds_softmax-a)))/(K+P0*(torch.exp(R*(-(preds_softmax-a)))-1)) - h) * beta
        loss = torch.mul(factor, loss.t())
        loss_sort, loss_idx = torch.sort(loss, descending=True)
        loss_sort = loss_sort[0, :20]
        if sort_loss:
            return loss_sort.mean()
        else:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
            return loss

class logistic_loss_weight(nn.Module):

    def __init__(self, alpha=0.3, num_classes=3, size_average=True):
        """
        logistic loss function
        :param num_classes:     the number of classes
        :param size_average:    the way of compute loss, default by mean value
        """
        super(logistic_loss_weight, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print("logistic_loss alpha = {}, please deassign it".format(alpha))
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            #self.alpha[1:] += (1-alpha)
            self.alpha[1:] += alpha

    def forward(self, preds, labels, loss_weight, ignore_index=-1):
        """
        logistic_loss compute loss
        :param preds: predict class. size:[B,N,C] or [B,C]    B is the batch size, N is the number of predict boxes, C is the number of class
        :param labels:  the ground truth's class label. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        keep_index = np.where(labels.cpu() != ignore_index)[0]
        preds = preds[keep_index, :]
        labels = labels[keep_index]
        preds_softmax = F.softmax(preds, dim=1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))   #  the same as nll_loss ( crossempty = log_softmax + nll )

        R = 10.9861
        P0 = 0.5
        K = 1.0
        beta = 1.5
        self.alpha = self.alpha.to(preds.device)
        a = self.alpha.gather(0, labels.view(-1))
        h = K*P0*torch.exp(R*(-(1-a)))/(K+P0*(torch.exp(R*(-(1-a)))-1))
        loss = (K*P0*torch.exp(R*(-(preds_softmax-a)))/(K+P0*(torch.exp(R*(-(preds_softmax-a)))-1)) - h) * beta
        loss_weight = torch.tensor(loss_weight).to(preds.device)
        loss = torch.mul(loss_weight, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss