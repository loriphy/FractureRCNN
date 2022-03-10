from __future__ import absolute_import
import os
import cv2
import ipdb
import matplotlib
from tqdm import tqdm
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
import numpy as np
import time
import torch
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
#import resource
import pandas as pd
import os.path
from utils.voc_eval import eval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    '''
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    '''
    valset = TestDataset(opt, split='val')
    val_dataloader = data_.DataLoader(valset,
                                      batch_size=1,
                                      num_workers=opt.test_num_workers,
                                      shuffle=False, \
                                      pin_memory=True
                                      )

    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = -1
    lr_ = opt.lr
    trainap=[]
    SEP = []
    LOSS = []
    for epoch in range(opt.epoch):
        print("the %dth epoch" % epoch)
        trainer.reset_meters()
        SE = []
        SP = []
        dt = 0
        for ii, (img, bbox_, label_, scale, imgpath) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            t = time.time()
            loss, se, sp = trainer.train_step(img, bbox, label, scale, imgpath[0], epoch, ii)
            dt += time.time() - t

            SE.append(se)
            SP.append(sp)

        print("dt", dt)
        SEP.append([np.mean(np.array(SE)), np.mean(np.array(SP))])
        if (epoch == opt.epoch - 1):
            eval_result = eval(val_dataloader, faster_rcnn, prin1=True, prin2=1)
        else:
            eval_result = eval(val_dataloader, faster_rcnn, prin1=False, prin2=2)
        # trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result),
                                                  str(trainer.get_meter_data()))
        print(log_info)
        trainap.append(eval_result)

        LOSSES = trainer.get_meter_data()
        LOSS.append([LOSSES['rpn_loc_loss'], LOSSES['rpn_cos_loss'], LOSSES['anchor_cls_loss'], LOSSES['roi_loc_loss'], LOSSES['roi_cls_loss'], LOSSES['total_loss']])

        if eval_result > best_map:
            best_map = eval_result
            best_path = trainer.save(best_map=best_map)

        if (epoch+1) == 20 or (epoch+1) == 40 or (epoch+1) == 60 or (epoch+1) == 80:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

def valandtest(**kwargs):
    opt._parse(kwargs)
    '''
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    '''
    valset = TestDataset(opt, split='val')
    val_dataloader = data_.DataLoader(valset,
                                      batch_size=1,
                                      num_workers=opt.test_num_workers,
                                      shuffle=False, \
                                      pin_memory=True
                                      )
    
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(opt.load_path)
    print('load pretrained model from %s' % opt.load_path)
    eval_result = eval(val_dataloader, faster_rcnn, prin1=True, prin2=1)
    #test_result = eval(test_dataloader, faster_rcnn, prin1=True, prin2=2)

def imgshow(img,gtbbox,predbbox,predscore,path,ii):
    img=cv2.imread(img)
    gtbbox=np.array(gtbbox[0])
    gtnum=gtbbox.shape[0]
    for i in range(gtnum):
        cv2.rectangle(img,(gtbbox[i,1],gtbbox[i,0]),(gtbbox[i,3],gtbbox[i,2]),(0,0,255),2)  # x1y1,x2y2,BGR
    predbbox=np.array(predbbox[0])
    prednum=predbbox.shape[0]
    if predbbox.any():
        for i in range(prednum):
            cv2.rectangle(img, (predbbox[i, 1], predbbox[i, 0]), (predbbox[i, 3], predbbox[i, 2]), (0, 255, 0), 2)
            cv2.putText(img, str(round(predscore[i],2)), (predbbox[i, 1], predbbox[i, 0] - 5), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
    #cv2.imshow('img with rectangle', img)
    #cv2.waitKey(0)
    save_path = './imgs/'+path+'/'
    cv2.imwrite(os.path.join(save_path, str(ii) + '.png'), img)


if __name__ == '__main__':
    import fire
    fire.Fire()
