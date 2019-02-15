'''
     TianChi competition: https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409106.5678.1.97f13501Kmu13E&raceId=231683
         Alibaba Cloud German AI Challenge 2018
     Author: zhangboshen
     Data: 20181212
'''

import h5py
import numpy as np
import os, sys
import matplotlib.pyplot as plt
#from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
import logging
import time
import random

import model
import sphere_model

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

import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


## Training hypes:
learning_rate = 0.00035
batch_size = 128
nepoch = 31
num_class = 17
save_dir = 'dualpath1818_res64'


try:
    os.makedirs(save_dir)
except OSError:
    pass


timer = time.time()

### loading data
base_dir = os.path.expanduser("./data")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')


s1_training = fid_training['sen1']
print ('s1_training.shape', s1_training.shape)
s2_training = fid_training['sen2']
print ('s2_training.shape',s2_training.shape)
label_training = fid_training['label']
print ('label_training.shape',label_training.shape)


s1_validation = fid_validation['sen1']
print ('s1_validation.shape', s1_validation.shape)
s2_validation = fid_validation['sen2']
print ('s2_validation.shape', s2_validation.shape)
label_validation = fid_validation['label']
print ('label_validation.shape',label_validation.shape)


timer = time.time() - timer
print('==> time to load data = %f (ms)' %(timer*1000))

s1_mean = np.load('./s1mean.npy')
s2_mean = np.load('./s2mean.npy')

s1_variance = np.load('./sq1mean.npy')
s2_variance = np.load('./sq2mean.npy')

s1_std = np.sqrt(s1_variance)
s2_std = np.sqrt(s2_variance)


class data_generator:   
    def create_train(s1_training, s2_training, label_training, batch_size, augument=True):
        HorizontalFlip = RandomHorizontalFlip(p=0.5)
        VerticalFlip = RandomVerticalFlip(p=0.5)
        erase = RandomErasure(p=0.5)
        resize = Resize(scalar = 2)
        while True:
            for start in range(0, len(label_training), batch_size):
                end = min(start + batch_size, len(label_training))
                
                X1_train_batch = np.array(s1_training[start:end])
                X2_train_batch = np.array(s2_training[start:end])

                X1_train_batch, X2_train_batch = resize(X1_train_batch, X2_train_batch)
###################################################
                X1_train_batch -= s1_mean     #np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456])
                X1_train_batch /= s1_std  #np.array([0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224])
###################################################				
                
###################################################
                X2_train_batch -= s2_mean #np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.485])
                X2_train_batch /= s2_std   #  np.array([0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.225, 0.229, 0.224, 0.225])
###################################################
                batch_labels = np.array(label_training[start:end])
                batch_labels = [np.argmax(one_hot) for one_hot in batch_labels]
                if augument:
                    X1_train_batch, X2_train_batch = HorizontalFlip(X1_train_batch,X2_train_batch)
                    X1_train_batch, X2_train_batch = VerticalFlip(X1_train_batch,X2_train_batch)
                    
                yield np.array(X1_train_batch, np.float32), np.array(X2_train_batch, np.float32), np.array(batch_labels)
            




class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, s1_training, s2_training, label_training, augument=True):

        self.HorizontalFlip = RandomHorizontalFlip(p=0.5)
        self.VerticalFlip = RandomVerticalFlip(p=0.5)
        self.erase = RandomErasure(p=0.5)
        self.resize = Resize(scalar = 2)
        self.s1_training = s1_training
        self.s2_training = s2_training
        self.label_training = label_training
        self.augument = augument

    def __getitem__(self, index):

        X1_train = np.array(self.s1_training[index])
        X2_train = np.array(self.s2_training[index])
        #print('################',X1_train.shape, X2_train.shape)
        X1_train, X2_train = self.resize(X1_train, X2_train)

        X1_train -= s1_mean     #np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456])
        X1_train /= s1_std

        X2_train -= s2_mean #np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.485])
        X2_train /= s2_std 

        label = np.array(self.label_training[index])
        #label = [np.argmax(one_hot) for one_hot in label]
        label = np.argmax(label)
        if self.augument:
            X1_train, X2_train = self.HorizontalFlip(X1_train, X2_train)
            X1_train, X2_train = self.VerticalFlip(X1_train, X2_train)
            X1_train, X2_train = self.erase(X1_train,X2_train)

        X1_train = np.array(X1_train, np.float32).transpose(2, 0, 1)
        X2_train = np.array(X2_train, np.float32).transpose(2, 0, 1)

        #print('X1_train',X1_train.shape, X2_train.shape)
        data = np.concatenate((X1_train, X2_train), axis=0)

        #X1_train = torch.from_numpy(X1_train.transpose(2, 0, 1))
        #X2_train = torch.from_numpy(X2_train.transpose(2, 0, 1))  # [H, W ,C] --->>> [C, H, W]

        #data = torch.cat([X1_train, X2_train], 0)
        label = np.array(label)

        return data, label 
    
    def __len__(self):
        return len(self.s1_training)


###----------------------------data aug-----------------------------###
def hflip(img):    
    img = img[:, range(img.shape[1]-1,-1,-1), :]
    return img

def vflip(img):
    img = img[range(img.shape[0]-1,-1,-1), :, :]
    return img


class RandomVerticalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p
    
    def __call__(self, img1, img2):
        if random.random()<self.p:
            img1 = vflip(img1)
            img2 = vflip(img2)
        return img1, img2


class RandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p
    
    def __call__(self, img1, img2):
        if random.random()<self.p:
            img1 = hflip(img1)
            img2 = hflip(img2)
        return img1, img2


class ToTensor(object):
    def __init__(self, range_=[0, 1]):
        self.range = range_[1] - range_[0]

    def factor(self, img):
        MaxValue = np.max(img)
        MinValue = np.min(img)
        k = self.range /(MaxValue-MinValue)
        return k

    def __call__(self, img1, img2):
        k1 = self.factor(img1)
        k2 = self.factor(img2)
        img1 = k1*(img1-np.min(img1))
        img2 = k2*(img2-np.min(img2))
        img1 = img1.transpose(2,0,1)
        img2 = img2.transpose(2,0,1)
        return img1, img2


class Normalize(object):
    def __init__(self, mean=[0, 0, 0], std=[0, 0 ,0]):
        self.mean = mean
        self.std = std

    def norma(self, img):
        for i in range(img.shape[0]):
            img[i,:,:] = (img[i,:,:] - self.mean[i])/self.std[i]
        return img

    def __call__(self, img1, img2):
        img1 = self.norma(img1)
        img2 = self.norma(img2)
        return img1, img2

        
class RandomErasure(object):
    def __init__(self, p=0.5):
        self.p = p        
        self.pad = np.random.normal(0, 1)

    def __call__(self, img1, img2):     
        if random.random()<self.p:   
            H = random.randint(1,4)
            W = random.randint(1,4)
            y = random.randint(0, img1.shape[1]-4)
            x = random.randint(0, img1.shape[2]-4)
            img1[y:y+H, x:x+W, :] = self.pad   
            img2[y:y+H, x:x+W, :] = self.pad   
        return img1, img2



class Resize(object):
    def __init__(self, scalar = 1):
        self.scalar = scalar

    def __call__(self, img1, img2):
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
        out1 = cv2.resize(img1, dsize=(h*self.scalar, w*self.scalar), interpolation = cv2.INTER_CUBIC)
        out2 = cv2.resize(img2, dsize=(h*self.scalar, w*self.scalar), interpolation = cv2.INTER_CUBIC)
        out1, out2 = np.array(out1), np.array(out2)
        return out1, out2

# split data into train, valid
#train_indexes = np.arange(len(s1_training))
#np.random.shuffle(train_indexes)
#valid_indexes = np.arange(len(s1_validation))
#np.random.shuffle(valid_indexes)
#train_indexes, valid_indexes = train_test_split(indexes, test_size=0.09886714727085479, random_state=8)

# create train and valid datagens
train_generator = data_generator.create_train(
    s1_training, s2_training, label_training, batch_size, augument=True)
validation_generator = data_generator.create_train(
    s1_validation, s2_validation, label_validation, batch_size//2, augument=False)


train_datasets = my_dataloader(s1_training, s2_training, label_training, augument=True)
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)#, collate_fn = my_collate_fn) # 8 workers may work faster

val_datasets = my_dataloader(s1_validation, s2_validation, label_validation, augument=True)
val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)#, collate_fn = my_collate_fn)

#netR = model.AlexNet_ft(num_classes = num_class)  
#netR = sphere_model.ResNet_ft(num_classes = num_class)  
#netR = model.VGG16_ft(num_classes = num_class)  
#netR = model.senet_DualPath(num_classes = num_class)  
#netR = model.ResNet_VLAD(num_classes = num_class)  
netR = model.ResNetSE_dualPath(num_classes = num_class)  
#netR.load_state_dict(torch.load('/data01/xiongfu/Tianchi/CloudGermanAI/resnet18_feat88_res64/netR_5e3_6.pth'))
netR.cuda()
#print(netR)


criterion = nn.CrossEntropyLoss().cuda()
#criterion = sphere_model.AngleLoss()
optimizer = optim.Adam(netR.parameters(), lr=learning_rate, betas = (0.9, 0.999), eps=1e-08, weight_decay = 5e-3)
#optimizer = optim.SGD(netR.parameters(),lr = learning_rate,momentum=0.9,weight_decay=1e-4)
#optimizer.load_state_dict(torch.load(os.path.join('/data05/zhangboshen/CloudGermanAI/resnet50_SE', 'optimizer_5e5_20.pth')))

scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')



## Traininig and eveluating
for epoch in range(nepoch):
    scheduler.step(epoch)
    print('================>>>>> Online epoch: #%d, lr=%f<<<<<================' %(epoch, scheduler.get_lr()[0]))
    # switch to train mode
    torch.cuda.synchronize()
    netR.train()
    train_loss_add = 0.0
    timer = time.time()
    counter_train = 0
    accuracy_train = 0
    #for step in range(int(len(s1_training)/batch_size)):
    for step, (data, label_) in enumerate(train_dataloaders):
        torch.cuda.synchronize()
      
        #img1, img2, label_ = train_generator.__next__()
        #img1 = img1.transpose(0,3,1,2)  # NHWC -->> NCHW
        #img2 = img2.transpose(0,3,1,2)  # NHWC -->> NCHW
        #img1, img2 = Variable(torch.from_numpy(img1).cuda()), Variable(torch.from_numpy(img2).cuda())   # NCHW format

        img1, img2, label_ = Variable(data[:,:8,:,:]).cuda(), Variable(data[:,8:,:,:]).cuda(), Variable(label_).cuda()
        label = label_.long()
        #label = Variable(label.cuda())


        # compute output
        optimizer.zero_grad()
        #estimation = netR(torch.cat([img1, img2], 1))
        estimation = netR(img1, img2)
        # computing accuracy
        for m in range(img1.shape[0]):
            pred_argmax = np.argmax(estimation.data.cpu().numpy()[m])  # N
            if pred_argmax == (label_.data.cpu().numpy())[m]:
                counter_train += 1       

        loss = criterion(estimation, label)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        train_loss_add = train_loss_add + (loss.item())*len(img1)

        #step = step + 1
        if(step % 50 == 0 ):
            print('epoch: %d,  step:  %d, loss: %f'%(epoch, step, loss.item()))

        if(step % 1001 == 0 ):
            print('epoch: %d,  step:  %d, loss: %f'%(epoch, step, loss.item()))
            print('prediction:', estimation[0:2].data.cpu().numpy())
            print('GT:', label[0:2].data.cpu().numpy())

    # time taken
    torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / len(s1_training)
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

    train_loss_add = train_loss_add / len(s1_training)
    print('mean error of 1 sample: %f, #train_indexes = %d' %(train_loss_add, len(s1_training)))

    accuracy_train = counter_train / (int(len(s1_training)/batch_size) * batch_size)
    print('training accuracy: %f, #train_indexes = %d' %(accuracy_train, len(s1_training)))


    # netR.load_state_dict(torch.load(os.path.join('/data05/zhangboshen/CloudGermanAI/resnet50_xf','netR_5e4_'+str(int(epoch+2))+'.pth')))
    test_loss_add = 0.0
    counter = 0   
    accuracy = 0

    # switch to evaluate mode
    torch.cuda.synchronize()
    netR.train()
    test_loss_add = 0.0
    timer = time.time()  
    #for i in range(int(len(s1_validation)/(batch_size//2))):
    for i, (data, label_) in enumerate(val_dataloaders):
        torch.cuda.synchronize()

        #img1, img2, label_ = validation_generator.__next__()
        #img1 = img1.transpose(0,3,1,2)  # NHWC -->> NCHW
        #img2 = img2.transpose(0,3,1,2)  # NHWC -->> NCHW
        #img1, img2 = Variable(torch.from_numpy(img1).cuda()), Variable(torch.from_numpy(img2).cuda())   # NCHW format
        img1, img2, label_ = Variable(data[:,:8,:,:]).cuda(), Variable(data[:,8:,:,:]).cuda(), Variable(label_).cuda()    
        label = label_.long()
        label = Variable(label.cuda())

        optimizer.zero_grad()

        estimation = netR(img1, img2)

        computing accuracy
        for m in range(img1.shape[0]):
            pred_argmax = np.argmax(estimation[0].data.cpu().numpy()[m])  # N
            if pred_argmax == label_.data.cpu().numpy()[m]:
                counter += 1                  

        loss = criterion(estimation, label)

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        test_loss_add = test_loss_add + (loss.item())*len(img1)

        if(i % 10 == 0 ):
            print('Evaluating   epoch: %d,  step:  %d, loss: %f'%(epoch, i, loss.item()))
 
        if(i % 201 == 0 ):
            print('Evaluating   epoch: %d,  step:  %d, loss: %f'%(epoch, i, loss.item()))
            print('prediction:', estimation[0:2].data.cpu().numpy())
            print('GT:', label[0:2].data.cpu().numpy())
    
    torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / (len(s1_validation)*3)
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

    test_loss_add = test_loss_add / (len(s1_validation))
    print('mean error of 1 sample: %f, #test_indexes = %d' %(test_loss_add, len(s1_validation)))

    accuracy = counter / (int(len(s1_validation)/batch_size) * batch_size)
    print('validation accuracy: %f, #test_indexes = %d' %(accuracy, len(s1_validation)))


    if (epoch % 2 == 0) and (epoch > 1):  
        torch.save(netR.state_dict(), '%s/netR_5e3_%d.pth' % (save_dir, epoch))
        torch.save(optimizer.state_dict(), '%s/optimizer_5e3_%d.pth' % (save_dir, epoch))
    # log
    logging.info('Epoch#%d: train error=%e, test error=%e, train acc=%e, validation acc=%e, lr = %f'
     %(epoch, train_loss_add, test_loss_add, accuracy_train, accuracy, scheduler.get_lr()[0]))





