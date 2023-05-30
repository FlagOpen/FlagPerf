### 模型信息
- 模型介绍
>The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.

- 论文
> [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)

- 模型代码来源
> https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

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



