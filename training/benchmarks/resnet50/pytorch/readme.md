
### 模型说明
ResNet50
Paper: "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385


### 原始模型的地址和精度
#### Resnet50原始模型 
`https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py`
> 该模型(V1.5)是对 Paper 中的Reset50V1版本做了小的改进。top1精度提升了约0.5%。
> 变化之处：bottleneck block, v1的第一个1x1 卷积用了 stride = 2。v1.5则使用了 stride = 2的3x3卷积来代替。

#### 精度信息

OPTIMIZER
Momentum (0.875)
Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we linearly scale the learning rate.
Learning rate schedule - we use cosine LR schedule
For bigger batch sizes (512 and up) we use linear warmup of the learning rate during the first couple of epochs according to Training ImageNet in 1 hour. Warmup length depends on the total training length.
Weight decay (WD)= 3.0517578125e-05 (1/32768).
We do not apply WD on Batch Norm trainable parameters (gamma/bias)
Label smoothing = 0.1

50 Epochs -> configuration that reaches 75.9% top1 accuracy
90 Epochs -> 90 epochs is a standard for ImageNet networks
250 Epochs -> best possible accuracy.

https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch


### 数据集下载

● 下载地址：https://image-net.org/

ImageNet2012 
|  Dataset   |Size|Checksum  |
|  ----  | - |----  |
|Training image (Task 1 & 2) | 138GB | MD5: ccaf1013018ac1037801578038d370da |
|Training image (Task 3 ) | 728MB | MD5: 1d675b47d978889d74fa0da5fadfb00e  |
|Validation images (all tasks)| 6.3GB | MD5: 29b22e2961454d5413ddabcf34fc5622 |
|Test images (all tasks)| 13GB| MD5: e1b8681fff3d63731c599df9b4b6fc02|
```
文件列表：
ILSVRC2012_img_train.tar
ILSVRC2012_img_train_t3.tar
ILSVRC2012_img_val.tar
ILSVRC2012_img_test_v10102019.tar
```

● 数据集格式转换：
https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

确保ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar与extract_ILSVRC.sh在同一目录。
```
sh extract_ILSVRC.sh
```

查看解压后的目录结构：tree -d -L 1

```
.
├── train
└── val
```
### Pytorch版本运行指南
#### 单机多卡运行：
1. FlagPerf/training/run_benchmarks/config/cluster_conf.py， 修改HOSTS为单个worker的ip
2. FlagPerf/training/run_benchmarks/config/test_conf.py中的CASES列表 添加RESNET50单机多卡配置: RESNET50_TORCH_DEMO_A100_1X8
运行：cd FlagPerf/training/run_benchmarks && python3 run.py


#### 多机多卡运行：
1. FlagPerf/training/run_benchmarks/config/cluster_conf.py， 修改HOSTS为多个worker的ip
2. FlagPerf/training/run_benchmarks/config/test_conf.py中的CASES列表 添加RESNET50多卡配置: RESNET50_TORCH_DEMO_A100_2X8

运行：cd FlagPerf/training/run_benchmarks && python3 run.py


### benchmark配置参数说明
- 基础参数(_base.py)
  
|  参数   | 类型|分类|说明  |
|  ----  | - |- |----  |
|arch| str |model | model name. should always be resnet50 for this case |
|train_batch_size| int |data | Total batch size for training |
|eval_batch_size | int |data | Total batch size for validating |
|learning_rate | float | lr_scheduler |initial learning rate for lr_scheduler|
|lr_decay_style | str |lr_scheduler | learning rate decay function |
|lr_decay_iters | int | lr_scheduler |number of iterations to decay LR over |
|weight_decay_rate | float |optimizer | learning rate decay function |
|momentum | float |optimizer | learning rate decay function |
|do_train | float |trainer | traininig flag |
|epoch | int |trainer | total number of iterations to train over all training runs |
|max_steps | int |trainer | Total number of training steps to perform|
|eval_interval_samples | int |trainer | number of training samples to run a evaluation once|
|target_accuracy | float |trainer |target accuracy to converge for training|
|dist_backend | str |Communication backend for distributed training on gpus. nccl/gloo|
|ddp_type | str |distributed |Distributed Data Parellel type|


- vendor参数(`<vendor>/<model>-<framework>/config/config_<accelerator>_<nnodes>x<nprocs>_.py`)
  
|  参数   | 类型|说明  |
|  ----  | - |----  |
|learning_rate | float | initial learning rate for lr_scheduler|
|dist_backend | str | Communication backend for distributed training on gpus. nccl/gloo |
|train_batch_size| int | Total batch size for training |
|eval_batch_size | int | Total batch size for validating |
|max_steps | int | Total number of training steps to perform|



### 许可证

本项目基于Apache 2.0 license。
本项目部分代码基于 https://github.com/pytorch/examples/tree/main/imagenet 实现。
