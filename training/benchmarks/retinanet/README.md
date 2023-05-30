### 模型信息
- Introduction

  RetinaNet is a one-stage object detection model that utilizes a focal loss function to address class imbalance during training. Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard negative examples. RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. The backbone is responsible for computing a convolutional feature map over an entire input image and is an off-the-self convolutional network. The first subnet performs convolutional object classification on the backbone's output; the second subnet performs convolutional bounding box regression. The two subnetworks feature a simple design that the authors propose specifically for one-stage, dense detection.

- Paper
[RetinaNet](https://arxiv.org/abs/1708.02002v2) 

- 模型代码来源
  https://github.com/pytorch/vision/tree/release/0.9/references/detection

### 数据集
#### 数据集下载地址
  COCO2017数据集
  COCO官网地址：https://cocodataset.org/


#### 预处理

这里以下载coco2017数据集为例，主要下载三个文件：
- 2017 Train images [118K/18GB]：训练过程中使用到的所有图像文件
- 2017 Val images [5K/1GB]：验证过程中使用到的所有图像文件
- 2017 Train/Val annotations [241MB]：对应训练集和验证集的标注json文件
都解压到coco2017文件夹下，可得到如下文件夹结构：

```bash
├── coco2017: # 数据集根目录
     ├── train2017: # 所有训练图像文件夹(118287张)
     ├── val2017: # 所有验证图像文件夹(5000张)
     └── annotations: # 对应标注文件夹
              ├── instances_train2017.json: # 对应目标检测、分割任务的训练集标注文件
              ├── instances_val2017.json: # 对应目标检测、分割任务的验证集标注文件
              ├── captions_train2017.json: # 对应图像描述的训练集标注文件
              ├── captions_val2017.json: # 对应图像描述的验证集标注文件
              ├── person_keypoints_train2017.json: # 对应人体关键点检测的训练集标注文件
              └── person_keypoints_val2017.json: # 对应人体关键点检测的验证集标注文件夹
```



#### Resnet50预训练权重 
 https://download.pytorch.org/models/resnet50-0676ba61.pth



### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- |
| Nvidia GPU | ✅       |
| 昆仑芯 XPU | N/A     |
| 天数智芯   | N/A     |


