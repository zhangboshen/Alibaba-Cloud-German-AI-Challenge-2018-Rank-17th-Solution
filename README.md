# Alibaba-Cloud-German-AI-Challenge-2018-Rank-17th-Solution
Alibaba Cloud German AI Challenge 2018, 17th place solution. https://tianchi.aliyun.com/competition/entrance/231683/introduction

---
## Development tools：
GPU: NVIDIA GTX-1080  
Code: Python3.6  
Framework: Pytorch1.0  
Optimizer: Adam  
Loss: CrossEntropyLoss and sphereFaceAngleLoss  
Schedule: StepLR with step_size=8, gamma=0.2   


---
## Solution：

- Validation set is the most important data for this task, because we’ve noticed that there is a shift between training set and validation set, while test set and validation set is very likely come from same distribution.

- We modified off-the-shelf networks to give us a higher feature map resolution, e.g. ResNet to give us a 4 times downsampling instead of 32. We doing this because original resolution is small compare with other images, higher feature map resolution helps with prediction accuracy. 

- For all our models, since input images are 18-channels collected from 2 sensors, we split input images channels into 8 and 10 channels respectively, two global pooling features were weighted sum before we get the final prediction.

- We’ve noticed that image resolution matters, e.g. for ResNet18 model, original size 32-32 gives us 0.835 accuracy, while 64-64 gives us 0.852 accuracy.

- Final results were obtained by 10 models ensemble, including ResNet18, ResNet34, VGG16, self-designed dual path SeResNet18, ResNet18 with VLAD, etc.

- False label will surely helps, because we’ve noticed that the accuracy is nearly fixed in all test set, this means that all of the test set come from same distribution. But we don’t have time for verifying this due to the deadline and some other stuffs.

- Different models requires different time, basically from 15 mins to 60 mins for 1 epoch. Maximum training epoch is 30 in our setting. Inference time is about 23fps to 80 fps.


## Usage：

*train.py* is training code, we use both training set and validation set for training, 
four *.npy* are mean and std computed through the whole training set.

*model.py* is the models that we used, *resnet.py* is a modified version of original resnet.py (torchvision), 
it consists of only 4 times downsampling instead of 32. *sphere_model.py* is used when we train our model 
with sphereFaceAngleLoss, *netvlad.py* is used in one of our models -- ResNet18 with VLAD.

*test.py* is testing code, it loads trained models and get the final prediction and gives us a submission.  


