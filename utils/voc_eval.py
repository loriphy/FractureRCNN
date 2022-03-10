import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import time
from tqdm import tqdm
import cv2
import pandas as pd

def voc_ap(rec, prec, use_07_metric=False):
    '''
    function: compute Ap according recall and precision
    input:
        rec: recall
        pre: precision
        use_07_metric: whether use the interpolate way in 2007, it takes 11 interpolation, but after 2010, seldom be used
    '''
    if use_07_metric:
        # 11 point metric
        Ap = 0.0
        for i in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= i) == 0:
                maxp = 0
            else:
                maxp = np.max(prec[rec >= i])
            Ap = Ap + maxp / 11.
    else:
        # use every point instead of only 11 point in use_07_metric
        # usually, we use this way to get more correct Ap after 2010
        # firstly, append flags at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # secondly, compute the precision in a small interval
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i -1], mpre[i])
        # thirdly, calculate area under PR curve, where x axis is recall and changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # sum (\delta recall) * prec
        Ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return Ap


def maxiou(gtbbox, predbbox):
    '''
    function: compute the iou between ground truth bbox and predict bbox
    input:
        gtbbox: ground truth bbox
        predbbox: predict bbox
    '''
    # intersection
    iymin = np.maximum(gtbbox[:, 0], predbbox[0])
    ixmin = np.maximum(gtbbox[:, 1], predbbox[1])
    iymax = np.minimum(gtbbox[:, 2], predbbox[2])
    ixmax = np.minimum(gtbbox[:, 3], predbbox[3])
    ih = np.maximum(iymax - iymin + 1., 0.)
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    inters = iw * ih
    # union
    union = (predbbox[2] - predbbox[0] + 1.) * (predbbox[3] - predbbox[1] + 1.) + \
            (gtbbox[:, 2] - gtbbox[:, 0] + 1.) * (gtbbox[:, 3] - gtbbox[:, 1] + 1.) - inters
    iou = inters / union
    ioumax = np.max(iou)
    ioumaxarg = np.argmax(iou)
    return ioumax, ioumaxarg


def eval_ap(imageid, npos, gtinfo, predbbox, predscore, iouthresh=0.5, use_07_metric=False):
    '''
    function: compute Ap according ground truth and predict bbox
    input:
        gtbbox: the ground truth bbox in a image
        gtlabel: the label of the correspond ground truth bbox
        predbbox: the predict bbox
        predscore: the predict score of every class of the correspond predict bbox
        iouthresh: when iou between ground truth and predict bbox greater or equal to iouthresh, we recognize it True Positive
        use_07_metric: whether use the interpolate way in 2007, it takes 11 interpolation, but after 2010, seldom be used
    '''
    # descending sort the predscore
    sorted_index = np.argsort(-predscore)
    sorted_predscore = np.sort(-predscore)
    predbbox = predbbox[sorted_index, :]
    imageid = imageid[sorted_index]
    # go down and make TP and FP
    num_pred = len(imageid)
    TP = np.zeros(num_pred)
    FP = np.zeros(num_pred)
    flag_CT = np.zeros(np.max(imageid)+1)
    test_CT = np.zeros(num_pred)
    for i in range(num_pred):
        GT = gtinfo[imageid[i]]
        bbpred = predbbox[i,:].astype(float)
        BBGT = GT['bbox'].astype(float)
        ioumax = -np.inf
        if BBGT.size > 0:
            ioumax, ioumaxarg = maxiou(BBGT, bbpred)
        # compute the number of TP and FP
        if ioumax >= iouthresh:
            if not GT['difficult'][ioumaxarg]:
                # if the ground truth have been detected, it can not be detected again
                if not GT['detection'][ioumaxarg]:
                    TP[i] = 1.
                    GT['detection'][ioumaxarg] = 1
                else:
                    FP[i] = 1.
        else:
            FP[i] = 1.
        flag_CT[imageid[i]] = 1
        test_CT[i] = np.sum(flag_CT)
    # compute recall
    FP = np.cumsum(FP)  # accumulation by bit
    TP = np.cumsum(TP)
    recall = TP/float(npos)
    # compute precision and note that avoid divide by zero
    # np.finfo(np.float64).eps is the greater than 0's infinitely small
    precision = TP/np.maximum(TP+FP, np.finfo(np.float64).eps)
    AP = voc_ap(recall, precision, use_07_metric)
    return recall, precision, AP, FP/test_CT


def eval(dataloader, faster_rcnn, prin1=False, prin2=0, classlabel=0, iouthresh=0.5):
    '''
    function: eval the model and get the recall, precision and AP
    input:
        dataloader: the data loader
        faster_rcnn: the model what we have trained
        prin1: a flag of whether print predbbox and gtbbox
        prin2: a flag of whether draw predbbox and gtbbox in image
        classlabel: the class that we want eval it
    '''
    prediction_time = 0. # record the prediction time
    npos = 0 # the number of ground truth
    GTinfo = {} # keep the infomation of ground truth
    imgid = []
    for i, (imgs, sizes, gtbbox, gtlabels, gtdifficults, imgpath) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        t = time.time()
        bbox, predlabels, score, rpn_loc_loss, rpn_cls_loss = faster_rcnn.predict(imgs, gtbbox, imgpath[0], gtlabels.reshape(-1), [sizes])
        prediction_time += (time.time() - t)
        gtlabels = gtlabels.reshape(-1)
        gtclass_index = np.where(gtlabels == classlabel)[0]
        gtbboxes = gtbbox.reshape(gtbbox.shape[0] * gtbbox.shape[1], 4)
        gtbboxes = np.array(gtbboxes)
        gtbboxes = gtbboxes[gtclass_index, :]
        gtdifficults = gtdifficults.reshape(-1)
        gtdifficults = np.array(gtdifficults[gtclass_index]).astype(np.bool)
        detection = [False] * len(gtclass_index) # a flag to label whether the object has been detect'
        npos = npos + sum(~gtdifficults)
        GTinfo[i] = {'bbox': gtbboxes, 'difficult': gtdifficults, 'detection': detection}
        for j in range(np.array(bbox[0]).shape[0]):
            imgid.append(i)
        resultshow(prin1, prin2, imgpath, gtbboxes, bbox, score, i, iouthresh)
        if i == 0:
            scores = score[0]
            bboxes = bbox[0]
        else:
            if bbox[0].shape[0] > 0:
                if bboxes.shape[0] > 0:
                    scores = np.concatenate((scores, score[0]), 0)
                    bboxes = np.concatenate((bboxes, bbox[0]), 0)
                else:
                    scores = score[0]
                    bboxes = bbox[0]
    recall, precision, AP, fpr = eval_ap(np.array(imgid), npos, GTinfo, np.array(bboxes), np.array(scores), iouthresh=iouthresh)
    PR = np.zeros((recall.shape[0], 4))
    PR[:, 0] = recall
    PR[:, 1] = precision
    PR[:, 2] = fpr
    PR[:, 3] = recall
    if prin2 > 0:
        print('the ap is:')
        print(AP)
        print('the ground truth num is:', npos)
        #'''
        data = pd.DataFrame(PR)
        if prin2 == 1:
            writer = pd.ExcelWriter('pr1.xlsx')
        else:
            writer = pd.ExcelWriter('pr2.xlsx')
        data.to_excel(writer, 'Sheet1', float_format='%0.8f')
        writer.save()
        writer.close()
        #'''
    #print("predict time:", prediction_time)
    return AP

def resultshow(prin1, prin2, imgpath,  gtbbox, predbbox, score, num, iouthresh=0.5):
    '''
    function: print of save the result of prediction
    '''
    if prin1:
        if (num + 1) % 1 == 2:
            print('the prebbox:')
            print(predbbox)
            print('the ground truth bbox:')
            print(gtbbox)
    predbbox = np.array(predbbox[0])
    positive_bool = np.zeros((predbbox.shape[0]))
    gt_flag = np.zeros((gtbbox.shape[0]))
    for i in range(predbbox.shape[0]):
        ioumax, ioumaxarg = maxiou(gtbbox, predbbox[i])
        if ioumax > iouthresh:
            if gt_flag[ioumaxarg] == 0:
                gt_flag[ioumaxarg] = 1
                positive_bool[i] = 1
    if prin2 == 1:
        imgshow(imgpath[0], gtbbox.astype(int), predbbox.astype(int), score, positive_bool, "valimg", num)
    if prin2 == 2:
        imgshow(imgpath[0], gtbbox.astype(int), predbbox.astype(int), score, positive_bool, "testimg", num)


def imgshow(imgpath, gtbbox, predbbox, predscore, positive_bool, path, ii):
    '''
    function: save the prediction result
    '''
    img = cv2.imread(imgpath)
    gtnum = gtbbox.shape[0]
    prednum = predbbox.shape[0]
    predscore = predscore[0]
    for i in range(gtnum):
        cv2.rectangle(img, (gtbbox[i, 1], gtbbox[i, 0]), (gtbbox[i, 3], gtbbox[i, 2]), (255, 144, 30), 2)
    if predbbox.any():
        for i in range(prednum):
            if positive_bool[i]:
                cv2.rectangle(img, (predbbox[i, 1], predbbox[i, 0]), (predbbox[i, 3], predbbox[i, 2]), (0, 225, 113), 2)
                cv2.putText(img, str(round(predscore[i], 2)), (predbbox[i, 1], predbbox[i, 0] - 5),
                            cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(0, 225, 113), thickness=1)
            else:
                cv2.rectangle(img, (predbbox[i, 1], predbbox[i, 0]), (predbbox[i, 3], predbbox[i, 2]), (0, 0, 255), 2)
                cv2.putText(img, str(round(predscore[i], 2)), (predbbox[i, 1], predbbox[i, 0] - 5),
                            cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)
    #for i in range(gtnum):
        #cv2.rectangle(img, (gtbbox[i, 1], gtbbox[i, 0]), (gtbbox[i, 3], gtbbox[i, 2]), (217, 180, 200), 2)  # x1y1,x2y2,BGR
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    save_path = './imgs/' + path + '/'
    cv2.imwrite(os.path.join(save_path, str(filename) + '.png'), img)




