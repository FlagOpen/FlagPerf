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

### 天数智芯 BI-V100 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本:  4.15.0-156-generic x86_64    
   - 加速卡驱动版本：3.0.0
   - Docker 版本：20.10.8
   - 训练框架版本：torch-1.10.2+corex.3.0.0
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件            | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s) |
| -------- | ------------------ | ---------- | ------- | -------  | ------- | --------------- |
| 单机1卡  | config_BI-V100x1x1 |     |      |    |    |     |    |    |    |        |        |
| 单机8卡  | config_BI-V100x1x8 | 5154.91    | 70.654    | 70.654   | 1156155    |2086.32          |


