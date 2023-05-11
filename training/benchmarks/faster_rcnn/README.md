### 模型信息
- Introduction

  State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. Faster RCNN is a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. Faster RCNN further merges RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.

- Paper
[Faster R-CNN](https://arxiv.org/abs/1506.01497) 

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
 https://download.pytorch.org/models/resnet50-19c8e357.pth



### 框架与芯片支持情况
|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | ✅       | N/A    | N/A         |
| 昆仑芯 XPU | N/A     | N/A    | N/A         |
| 天数智芯   | N/A     | N/A    | N/A         |


