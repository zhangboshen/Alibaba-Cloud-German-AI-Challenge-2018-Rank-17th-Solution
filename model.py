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


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ResNet_ft(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_ft, self).__init__()
        
        modelPreTrain = resnet.resnet18(pretrained=True)
        self.model = modelPreTrain
        modelPreTrain2 = resnet.resnet18(pretrained=True)
        self.model2 = modelPreTrain2
        
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
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        #self.linear = nn.Linear(2048, num_classes)
        #init.xavier_normal_(self.linear.weight)
        
    
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
        out = self.bn1(out)
        out1 = self.classifier1(out)
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
        out = self.bn2(out)
        out2 = self.classifier2(out)

        out = (out1 + out2)/2.
        return out



class ResNet_VLAD(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_VLAD, self).__init__()
        
        modelPreTrain = resnet.resnet18(pretrained=True)
        self.model = modelPreTrain
        modelPreTrain2 = resnet.resnet18(pretrained=True)
        self.model2 = modelPreTrain2
        
        self.num_clusters = 32

        self.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
        self.conv1.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
        self.conv1.weight.data[:,6:8,:,:] = self.model.conv1.weight.data[:,0:2,:,:]

        self.conv2 = nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,6:9,:,:] = self.model.conv1.weight.data
        self.conv2.weight.data[:,9,:,:] = self.model.conv1.weight.data[:,0,:,:]
        
        self.vlad1 = netvlad.NetVLAD(num_clusters=self.num_clusters, dim=512, alpha=1.0)
        self.vlad2 = netvlad.NetVLAD(num_clusters=self.num_clusters, dim=512, alpha=1.0)
        #self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn1 = nn.BatchNorm1d(512*self.num_clusters)
        self.bn2 = nn.BatchNorm1d(512*self.num_clusters)
        self.classifier1 = nn.Linear(512*self.num_clusters, num_classes)
        self.classifier2 = nn.Linear(512*self.num_clusters, num_classes)
        #self.linear = nn.Linear(2048, num_classes)
        #init.xavier_normal_(self.linear.weight)
        
    
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
        out = self.vlad1(out)   # torch.Size([8, 2048, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        out1 = self.classifier1(out)
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
        out = self.vlad2(out)   # torch.Size([8, 2048, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out2 = self.classifier2(out)

        out = (out1 + out2)/2.
        return out



# class ResNet_ft_18Channel(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet_ft_18Channel, self).__init__()
        
#         modelPreTrain = resnet.resnet50(pretrained=True)
#         self.model = modelPreTrain
        
#         self.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=1, padding=3, bias=False)
#         self.conv1.weight.data[:,0:3,:,:] = self.model.conv1.weight.data
#         self.conv1.weight.data[:,3:6,:,:] = self.model.conv1.weight.data
#         self.conv1.weight.data[:,6:9,:,:] = self.model.conv1.weight.data
#         self.conv1.weight.data[:,9:12,:,:] = self.model.conv1.weight.data
#         self.conv1.weight.data[:,12:15,:,:] = self.model.conv1.weight.data
#         self.conv1.weight.data[:,15:18,:,:] = self.model.conv1.weight.data
#         self.relu = nn.ReLU(inplace=True)
#         #self.bn1 = nn.BatchNorm2d(3)
#         #self.bn = nn.BatchNorm2d(2048)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.classifier = ClassBlock(2048, num_classes)
#         #self.linear = nn.Linear(2048, num_classes)
#         #init.xavier_normal_(self.linear.weight)
        
    
#     def forward(self, x):
#         out = self.conv1(x)
#         #out = self.model.conv1(x)
#         out = self.model.bn1(out)
#         out = self.model.relu(out)
#         out = self.model.maxpool(out)
#         out = self.model.layer1(out)
#         out = self.model.layer2(out)
#         out = self.model.layer3(out)
#         out = self.model.layer4(out)
#         print(out.size())
#         out = self.avg_pool(out)   # torch.Size([8, 2048, 1, 1])
#         #out = self.bn(out)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
      
#         return out


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







class AlexNet_ft(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet_ft, self).__init__()
        
        modelPreTrain = models.alexnet(pretrained=False)
        modelPreTrain.load_state_dict(torch.load('./weights/alexnet-owt-4df8aa71.pth'),False)
        self.model = modelPreTrain
        modelPreTrain2 = models.alexnet(pretrained=False)
        modelPreTrain2.load_state_dict(torch.load('./weights/alexnet-owt-4df8aa71.pth'),False)
        self.model2 = modelPreTrain2
        
        self.conv1 = nn.Conv2d(8, 64, kernel_size=11, stride=2, padding=2)
        self.conv1.weight.data[:,0:3,:,:] = self.model.features[0].weight.data
        self.conv1.weight.data[:,3:6,:,:] = self.model.features[0].weight.data
        self.conv1.weight.data[:,6:8,:,:] = self.model.features[0].weight.data[:,0:2,:,:]

        self.conv2 = nn.Conv2d(10, 64, kernel_size=11, stride=2, padding=2)
        self.conv2.weight.data[:,0:3,:,:] = self.model.features[0].weight.data
        self.conv2.weight.data[:,3:6,:,:] = self.model.features[0].weight.data
        self.conv2.weight.data[:,6:9,:,:] = self.model.features[0].weight.data
        self.conv2.weight.data[:,9,:,:] = self.model.features[0].weight.data[:,0,:,:]
        #self.conv2 = conv3x3(3,16)
        #self.relu = nn.ReLU(inplace=True)
        #self.bn1 = nn.BatchNorm2d(3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier1 = nn.Linear(256, num_classes)
        self.classifier2 = nn.Linear(256, num_classes)
        #self.linear = nn.Linear(2048, num_classes)
        #init.xavier_normal_(self.linear.weight)
        
    
    def forward(self, x1, x2):
        out = self.conv1(x1)
        
        out = self.model.features[1](out)  # nn.ReLU(inplace=True),
        out = self.model.features[3](out)  # nn.Conv2d(64, 192, kernel_size=5, padding=2),
        out = self.model.features[4](out)  # nn.ReLU(inplace=True),
        out = self.model.features[5](out)  # nn.MaxPool2d(kernel_size=3, stride=2),
        out = self.model.features[6](out)  # nn.Conv2d(192, 384, kernel_size=3, padding=1),
        out = self.model.features[7](out)  # nn.ReLU(inplace=True),
        out = self.model.features[8](out)  # nn.Conv2d(384, 256, kernel_size=3, padding=1),
        out = self.model.features[9](out)  # nn.ReLU(inplace=True),
        out = self.model.features[10](out)  # nn.Conv2d(256, 256, kernel_size=3, padding=1),
        out = self.model.features[11](out)  # nn.ReLU(inplace=True),

        out = self.avg_pool(out)   # torch.Size([8, 2048, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        out1 = self.classifier1(out)
        #out = self.linear(out)
        

        out = self.conv2(x2)
        #out = self.model.conv1(x)
        out = self.model2.features[1](out)  # nn.ReLU(inplace=True),
        out = self.model2.features[3](out)  # nn.Conv2d(64, 192, kernel_size=5, padding=2),
        out = self.model2.features[4](out)  # nn.ReLU(inplace=True),
        out = self.model2.features[5](out)  # nn.MaxPool2d(kernel_size=3, stride=2),
        out = self.model2.features[6](out)  # nn.Conv2d(192, 384, kernel_size=3, padding=1),
        out = self.model2.features[7](out)  # nn.ReLU(inplace=True),
        out = self.model2.features[8](out)  # nn.Conv2d(384, 256, kernel_size=3, padding=1),
        out = self.model2.features[9](out)  # nn.ReLU(inplace=True),
        out = self.model2.features[10](out)  # nn.Conv2d(256, 256, kernel_size=3, padding=1),
        out = self.model2.features[11](out)  # nn.ReLU(inplace=True),
        #print(out.shape)
        out = self.avg_pool(out)   # torch.Size([8, 2048, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out2 = self.classifier2(out)

        out = (out1 + out2)/2.
        return out



class VGG16_ft(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_ft, self).__init__()
        
        modelPreTrain = models.vgg16(pretrained=True)
        #modelPreTrain.load_state_dict(torch.load('./weights/alexnet-owt-4df8aa71.pth'),False)
        self.model = modelPreTrain
        modelPreTrain2 = models.vgg16(pretrained=True)
        #modelPreTrain2.load_state_dict(torch.load('./weights/alexnet-owt-4df8aa71.pth'),False)
        self.model2 = modelPreTrain2
        
        tmp = self.model.features[0]
        self.model.features[0] = torch.nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.features[0].weight.data[:,0:3,:,:] = tmp.weight.data
        self.model.features[0].weight.data[:,3:6,:,:] = tmp.weight.data
        self.model.features[0].weight.data[:,6:8,:,:] = tmp.weight.data[:,0:2,:,:]

        self.vgg1_modified = self.model
        self.vgg1_modified.features = nn.Sequential(*[self.model.features[i] for i in (0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29)])

        self.model2.features[0] = torch.nn.Conv2d(10, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model2.features[0].weight.data[:,0:3,:,:] = tmp.weight.data
        self.model2.features[0].weight.data[:,3:6,:,:] = tmp.weight.data
        self.model2.features[0].weight.data[:,6:9,:,:] = tmp.weight.data
        self.model2.features[0].weight.data[:,9,:,:] = tmp.weight.data[:,0,:,:]

        self.vgg2_modified = self.model2
        self.vgg2_modified.features = nn.Sequential(*[self.model2.features[i] for i in (0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29)])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        #self.linear = nn.Linear(2048, num_classes)
        #init.xavier_normal_(self.linear.weight)
        
    
    def forward(self, x1, x2):

        out = self.vgg1_modified.features(x1) 
        out = self.avg_pool(out)   
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        out1 = self.classifier1(out)
        #out = self.linear(out)
        

        out = self.vgg2_modified.features(x2)
        #print(out.shape)
        out = self.avg_pool(out)  
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out2 = self.classifier2(out)

        out = (out1 + out2)/2.
        return out



# class senet_DualPath(nn.Module):
#     def __init__(self, num_classes):
#         super(senet_DualPath, self).__init__()
        
#         #modelPreTrain = senet.se_resnext50_32x4d(num_classes=num_classes)
#         modelPreTrain = senet.se_resnext50_32x4d_MLFN(num_classes=num_classes)
        
#         self.model = modelPreTrain
#         #modelPreTrain2 = senet.se_resnext50_32x4d(num_classes=num_classes)
#         modelPreTrain2 = senet.se_resnext50_32x4d_MLFN(num_classes=num_classes)
#         self.model2 = modelPreTrain2
        
#         tmp = self.model.layer0[0]
#         self.model2.layer0[0] = nn.Conv2d(10, 64, kernel_size=7, stride=2,
#                                     padding=3, bias=False)
#         self.model2.layer0[0].weight.data[:,0:3,:,:] = tmp.weight.data[:,0:3,:,:]
#         self.model2.layer0[0].weight.data[:,3:6,:,:] = tmp.weight.data[:,0:3,:,:]
#         self.model2.layer0[0].weight.data[:,6:9,:,:] = tmp.weight.data[:,0:3,:,:]
#         self.model2.layer0[0].weight.data[:,9,:,:] = tmp.weight.data[:,0,:,:]

        
        
    
#     def forward(self, x1, x2):

#         out1 = self.model(x1) 
#         #out = self.avg_pool(out)   
#         #out = out.view(out.size(0), -1)
#         #out = self.bn1(out)
#         #out1 = self.classifier1(out)
#         #out = self.linear(out)
        

#         out2 = self.model2(x2)
#         #print(out.shape)
#         #out = self.avg_pool(out)  
#         #out = out.view(out.size(0), -1)
#         #out = self.bn2(out)
#         #out2 = self.classifier2(out)

#         out = (out1 + out2)/2.
#         return out



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
        
        