### 模型信息
- 模型介绍
>EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. 

- 论文
> [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

- 模型代码来源
This case includes code from the BSD 3-Clause License open source project at https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
<br>

### 数据集
> ImageNet_1k_2012数据集

> ImageNet官网地址：https://www.image-net.org/challenges/LSVRC/2012/

- 预处理
> 无需预处理


### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | ✅ |N/A  |N/A|



