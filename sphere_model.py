import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from torchvision import models
from torch.nn import init
import resnet
from torch.autograd import Variable
import sys,os
import netvlad
import time
import senet


from torch.nn import Parameter
import math

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class ResNet_ft(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_ft, self).__init__()
        
        modelPreTrain = resnet.resnet18(pretrained=True)
        self.model = modelPreTrain
        modelPreTrain2 = resnet.resnet18(pretrained=True)
        self.model2 = modelPreTrain2
        
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
        self.conv1.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
        self.conv1.weight.data[:,6:8,:,:] = self.model.conv1.weight.data[:,0:2,:,:]

        self.conv2 = nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,6:9,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,9,:,:] = self.model.conv1.weight.data[:,0,:,:]
        #self.conv2 = conv3x3(3,16)
        #self.relu = nn.ReLU(inplace=True)
        #self.bn1 = nn.BatchNorm2d(3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        #self.classifier1 = nn.Linear(512, num_classes)
        #self.classifier2 = nn.Linear(512, num_classes)
        self.classifier1 = AngleLinear(512, self.num_classes)
        self.classifier2 = AngleLinear(512, self.num_classes)
        
        
    
    def forward(self, x1, x2):
        out = self.conv1(x1)
        #out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        #out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        #print(out.shape)
        out = self.avg_pool(out)   # torch.Size([8, 2048, 1, 1])
        out = out.view(out.size(0), -1)
        out1 = self.bn1(out)
        #out1 = self.classifier1(out)
        #out = self.linear(out)
        

        out = self.conv2(x2)
        #out = self.model.conv1(x)
        out = self.model2.bn1(out)
        out = self.model2.relu(out)
        #out = self.model2.maxpool(out)
        out = self.model2.layer1(out)
        out = self.model2.layer2(out)
        out = self.model2.layer3(out)
        out = self.model2.layer4(out)
        #print(out.shape)
        out = self.avg_pool(out)   # torch.Size([8, 2048, 1, 1])
        out = out.view(out.size(0), -1)
        out2 = self.bn2(out)

        out = out1 + out2
        out = self.classifier2(out)


        #out = (out1 + out2)/2.
        return out



'''
SE NET
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  



class ResNetSE_dualPath(nn.Module):
    def __init__(self, num_classes):
        super(ResNetSE_dualPath, self).__init__()
        
        modelPreTrain50 = resnet.resnet18(pretrained=True)
        modelPreTrain50_2 = resnet.resnet18(pretrained=True)
        self.model = modelPreTrain50
        self.model2 = modelPreTrain50_2
        
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
        self.conv1.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
        self.conv1.weight.data[:,6:8,:,:] = self.model.conv1.weight.data[:,0:2,:,:]

        self.conv2 = nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,6:9,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,9,:,:] = self.model.conv1.weight.data[:,0,:,:]

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.SE1_1 = SELayer(64)
        self.SE2_1 = SELayer(64)
        self.SE1_2 = SELayer(128)
        self.SE2_2 = SELayer(128)
        self.SE1_3 = SELayer(256)
        self.SE2_3 = SELayer(256)
        self.SE1_4 = SELayer(512)
        self.SE2_4 = SELayer(512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = nn.Linear(512, num_classes)
        self.linear2 = nn.Linear(512, num_classes)
        #init.constant_(self.conv1.bias, 0)
        #init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
        #self.weight = Parameter(torch.Tensor(num_classes))
    
    def forward(self, x1, x2): 
        #n, c, h, w = x.size()
        
        #x1 = x[:,0:3,:,:]  # RGB
        #x2 = x[:,3:4,:,:]  # Y
        #x2 = x2.expand(n,3,h,w)
         
        #x2 = self.conv1(x2) 

        #x = torch.cat([x1,x2],0)       
        out1 = self.conv1(x1)
        #out1 = self.model.conv1(x1)
        out1 = self.model.bn1(out1)
        out1 = self.model.relu(out1)
        #out1 = self.model.maxpool(out1)

        out2 = self.conv2(x2)
        #out2 = self.model2.conv1(x2)
        out2 = self.model2.bn1(out2)
        out2 = self.model2.relu(out2)
        #out2 = self.model2.maxpool(out2)



        out1 = self.model.layer1(out1)
        out2 = self.model2.layer1(out2)
        out1,out2 = self.SE2_1(out2)*out1, self.SE1_1(out1)*out2  

        out1 = self.model.layer2(out1)
        out2 = self.model2.layer2(out2)
        out1,out2 = self.SE2_2(out2)*out1, self.SE1_2(out1)*out2  

        out1 = self.model.layer3(out1)
        out2 = self.model2.layer3(out2)
        out1,out2 = self.SE2_3(out2)*out1, self.SE1_3(out1)*out2  

        out1 = self.model.layer4(out1)
        out2 = self.model2.layer4(out2)
        out1,out2 = self.SE2_4(out2)*out1, self.SE1_4(out1)*out2  

        #print('#######', out1.size(), out2.size())

        out1 = self.avg_pool(out1)
        out1 = self.bn1(out1)   
        

        out2 = self.avg_pool(out2)
        out2 = self.bn2(out2)

        #print('#############', out1.size(), out2.size())
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        
        out1 = self.linear1(out1)
        out2 = self.linear2(out2)
######################################
        return (out1 + out2)/2




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    timer = time.time()
    net = ResNetSE_dualPath(17)
    print(net)
    #timer = time.time()
    y = net(Variable(torch.randn(4,8,32,32)),Variable(torch.randn(4,10,32,32)))
    #y = net(Variable(torch.randn(4,18,32,32)))
    print(y.size())

    #netR = model.ResNet_ft(num_classes = num_class)  # ResNet101
    #net.load_state_dict(torch.load('/data01/xiongfu/Tianchi/CloudGermanAI/resnet18_concat_train_xf/netR_5e3_20.pth'))

    #print(y2.size())
    timer = time.time() - timer
    print('time comsuming(sec): ',timer)  # cuda:20s, w/o cuda:80s

    #for name, value in net.named_parameters():
        #print(name)
    #for name, value in net.named_parameters():
        #if name[0:5] == 'model' and name[15:17] == 'bn':
            #print(name)
            #value.requires_grad = False
        
        