# ViT

## 模型信息

- 模型介绍
Vision Transformer (ViT) 是一种将Transformer应用在图像分类的模型。
[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf) (ICLR 2021)

ViT的结构是基于标准Transformer，将图片划分为patch作为Transformer的输入。

- 模型代码来源

<https://github.com/huggingface/pytorch-image-models>

## 数据集

- 数据集下载地址

ImageNet-1k
Source: <http://image-net.org/challenges/LSVRC/2012/index>
Paper: "ImageNet Large Scale Visual Recognition Challenge" - <https://arxiv.org/abs/1409.0575>

### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | ✅ |N/A  |N/A|

