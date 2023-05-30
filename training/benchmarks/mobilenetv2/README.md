### 模型信息
- 模型介绍
>MobileNet-v2 is a convolutional neural network that is 53 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. 
>Refer to Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4510-4520). IEEE.

- 模型代码来源
> https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

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
| 天数智芯GPU｜✅ |N/A  |N/A|


