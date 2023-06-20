### 模型信息
- Introduction
<br>
Mask R-CNN is simple, flexible, and general framework for object instance segmentation. It efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.
<br>
Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing users to estimate human poses in the same framework. The authors show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. The authors hope their simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition.
<br>

- Paper
[Mask R-CNN](https://arxiv.org/abs/1703.06870) 

- 模型代码来源
This case includes code from the BSD 3-Clause License open source project [torchvision] at https://github.com/pytorch/vision/tree/release/0.9/references/detection

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
<br>

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



### 模型checkpoint 
预训练权重下载地址（下载后放入data_dir文件夹中）

#### Resnet50预训练权重 
 https://download.pytorch.org/models/resnet50-0676ba61.pth 
 (注意，下载预训练权重后要重命名， 比如在train.py中读取的是resnet50.pth文件，不是resnet50-0676ba61.pth)

### 框架与芯片支持情况
|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | ✅       | N/A    | N/A         |
| 昆仑芯 XPU | N/A     | N/A    | N/A         |
| 天数智芯   | N/A     | N/A    | N/A         |



