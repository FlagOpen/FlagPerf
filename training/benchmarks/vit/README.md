# ViT

## 模型信息

- 模型介绍
Vision Transformer (ViT) 是一种将Transformer应用在图像分类的模型。
[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf) (ICLR 2021)

ViT的结构:

基于标准Transformer，将图片划分为patch作为Transformer的输入。

## 数据集

- 数据集下载地址

<https://bd.bcebos.com/klx-pytorch-ipipe-bd/xacc_dependencies/migrated_from_important/xmlir/vit/datasets/xmlir/train.tar.gz>

- 预处理
  
> 无需预处理

## 模型checkpoint

<https://github.com/huggingface/pytorch-image-models>

### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | ✅ |N/A  |N/A|

