### 模型信息
- 模型介绍
>EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. 
>Refer to EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

- 模型代码来源
> https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

### 数据集
- 数据集下载地址
> `https://image-net.org/download.php`  (Imagenet2012 1K)

- 预处理
> 无需预处理 


### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | ✅ |N/A  |N/A|



