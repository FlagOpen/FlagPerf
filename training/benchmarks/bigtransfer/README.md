### 模型信息
- Introduction

  Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision. BIT(Big Transfer) revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. By combining a few carefully selected components, and transferring using a simple heuristic, BIT achieve strong performance on over 20 datasets. 

- Paper
[Big Transfer](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500477.pdf) 

- 模型代码来源
[Google(author)](https://github.com/google-research/big_transfer) 

### 数据集
#### 数据集下载地址
  ImageNet_1k_2012数据集
  ImageNet官网地址：https://www.image-net.org/challenges/LSVRC/2012/


#### 预处理

按照原作者开源代码库的方式，对训练/验证集分别做如下处理：
- 训练集

  ```python
  train_tx = tv.transforms.Compose([
          tv.transforms.Resize((precrop, precrop)),
          tv.transforms.RandomCrop((crop, crop)),
          tv.transforms.RandomHorizontalFlip(),
          tv.transforms.ToTensor(),
          tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ])
  ```

  

- 验证集

  ```python
  val_tx = tv.transforms.Compose([
          tv.transforms.Resize((crop, crop)),
          tv.transforms.ToTensor(),
          tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ])
  ```

  

#### 迁移学习预训练权重 
https://storage.googleapis.com/bit_models/BiT-M-R152x{}.npz

其中{}可填写2或4。本标准case以BiT-M-R152x2为准。BiT-M-R152x4的训练需要添加下述优化中的任意一个：

1. 更改batchsize为1或者2
2. 使用torch 2.0 FSDP而非DDP包裹模型。注意：采用优化2需要同时更改启动镜像为torch2.0版本对应镜像



### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- |
| Nvidia GPU | ✅       |
| 昆仑芯 XPU | N/A     |
| 天数智芯   | N/A     |


