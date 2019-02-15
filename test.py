
import h5py
import numpy as np
import pandas as pd
import os, sys
#from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import logging
import time
from tqdm import tqdm

import model

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import cv2
import sphere_model

import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result_sav_dir = './D_results/'

model_file = '/data01/xiongfu/Tianchi/CloudGermanAI/resnet18_feature88_VLAD_concat_train_xf/'

csvFileName = 'D_TTA_h_v_flip_resnet18_feat88_VLAD_res32_epoch30_val_980_a820_.csv'

### Network def
#netR = model.ResNet_ft(num_classes = 17)  
#netR = model.VGG16_ft(num_classes = 17)  
#netR = model.AlexNet_ft(num_classes = 17)  
#netR = model.senet_DualPath(num_classes = 17)  
netR = model.ResNet_VLAD(num_classes = 17)  
#netR = model.ResNetSE_dualPath(num_classes = 17)  
netR.load_state_dict(torch.load(model_file + 'netR_5e3_30.pth'))
netR.cuda()
netR = netR.eval()
#print(netR)


base_dir = os.path.expanduser("./data")
path_test = os.path.join(base_dir, 'round2_test_b_20190211.h5')

fid_test = h5py.File(path_test,'r')


s1_test = fid_test['sen1']
print ('s1_test.shape', s1_test.shape)
s2_test = fid_test['sen2']
print ('s2_test.shape',s2_test.shape)

s1_mean = np.load('./s1mean.npy')
s2_mean = np.load('./s2mean.npy')

s1_variance = np.load('./sq1mean.npy')
s2_variance = np.load('./sq2mean.npy')

s1_std = np.sqrt(s1_variance)
s2_std = np.sqrt(s2_variance)




##################   TTA    ##########
def hflip(img):    
    img = img[:, :, range(img.shape[2]-1,-1,-1), :]
    return img

def vflip(img):
    img = img[:, range(img.shape[1]-1,-1,-1), :, :]
    return img


def Resize(img1, img2, scalar):
    # n,h,w,c = img1.shape 
    # out1 = []
    # out2 = []
    # for i in range(n):
    #     temp1 = cv2.resize(img1[i], dsize=(h*self.scalar, w*self.scalar), interpolation = cv2.INTER_CUBIC)
    #     temp2 = cv2.resize(img2[i], dsize=(h*self.scalar, w*self.scalar), interpolation = cv2.INTER_CUBIC)
    #     out1.append(temp1)
    #     out2.append(temp2)
    # out1, out2 = np.array(out1), np.array(out2)
    # return out1, out2
    h,w,c = img1.shape 
    out1 = cv2.resize(img1, dsize=(h*scalar, w*scalar), interpolation = cv2.INTER_CUBIC)
    out2 = cv2.resize(img2, dsize=(h*scalar, w*scalar), interpolation = cv2.INTER_CUBIC)
    out1, out2 = np.array(out1), np.array(out2)
    return out1, out2


def GetPredicion():
    pred = np.zeros((len(s1_test),17),dtype = np.float32)
    predArgmax = np.zeros((len(s1_test)),dtype = np.float32)
    predOneHot = np.zeros((len(s1_test),17),dtype = np.int64)

    for i in tqdm(range(len(s1_test))):
        img1, img2 = np.array(s1_test[i]), np.array(s2_test[i])
        #img1, img2 = Resize(img1, img2, scalar=2)

        img1 -= s1_mean #np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456])
        img1 /= s1_std #np.array([0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224])
        img2 -= s2_mean #np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.485])
        img2 /= s2_std #np.array([0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.225, 0.229, 0.224, 0.225])
        img1, img2 = np.expand_dims(img1, 0), np.expand_dims(img2, 0)
        # TTA
        img1_, img2_ = hflip(img1), hflip(img2)
        img1__, img2__ = vflip(img1), vflip(img2)

        img1, img2 = img1.transpose(0,3,1,2), img2.transpose(0,3,1,2)   # NHWC -->> NCHW
        img1_, img2_ = img1_.transpose(0,3,1,2), img2_.transpose(0,3,1,2)   # NHWC -->> NCHW
        img1__, img2__ = img1__.transpose(0,3,1,2), img2__.transpose(0,3,1,2)   # NHWC -->> NCHW

        img1, img2 = Variable(torch.from_numpy(img1).float().cuda()), Variable(torch.from_numpy(img2).float().cuda())   # NCHW format
        img1_, img2_ = Variable(torch.from_numpy(img1_).float().cuda()), Variable(torch.from_numpy(img2_).float().cuda())   # NCHW format
        img1__, img2__ = Variable(torch.from_numpy(img1__).float().cuda()), Variable(torch.from_numpy(img2__).float().cuda())   # NCHW format

        estimation1 = netR(img1, img2)
        estimation2 = netR(img1_, img2_)
        estimation3 = netR(img1__, img2__)
        #estimation = netR(torch.cat([img1, img2], 1))

        pred[i] = (estimation1.data.cpu().numpy() + estimation2.data.cpu().numpy() + estimation3.data.cpu().numpy())/3.0

        predArgmax[i] = np.argmax(pred[i])

        predOneHot[i, np.argmax(pred[i])] = 1

    return pred, predArgmax, predOneHot




pred, predArgmax, predOneHot = GetPredicion()


predDataFrame = pd.DataFrame(predOneHot)

np.save(result_sav_dir + csvFileName[:-4] + '.npy', pred)
predDataFrame.to_csv(result_sav_dir + csvFileName, header=None, index=False)

