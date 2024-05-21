### 模型信息

#### 模型介绍
Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.
#### 模型代码来源
| repo    | commmit_id  | date |
|  ----  | ----  |----  |
| [detr](https://github.com/facebookresearch/detr) |3af9fa878e73b6894ce3596450a8d9b89d918ca9 |2023-2-7 05:12:31|

This repository includes code is licensed under the Apache License, Version 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

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