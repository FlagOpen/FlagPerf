### 模型信息
- 模型介绍
>EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. 

- 论文
> [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

- 模型代码来源
> https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

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



