# 模型架构

ResNet的总体网络架构如下：
[链接](https://arxiv.org/pdf/1512.03385.pdf)

# 数据集
使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─validation_preprocess # 评估数据集
```

# 快速入门

## 代码结构

```shell
.
└──config
  ├── README.md
  ├── config                                        # 参数配置
    ├── config.sh                                   # 配置文件
    ├── resnet50_imagenet2012_Boost_config.yaml     # 高性能版本：性能提高超过10%而精度下降少于1%
  ├── common                                        # 公共函数文件
  ├── code                                          # 训练代码
  └── run_pretraining.sh                            # 训练网络
```

## 脚本参数

在配置文件中可以同时配置训练参数和评估参数。

- 配置ResNet18、ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":2,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

### 用法
分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

训练结果临时文件保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果

运行单卡用例时如果想更换运行卡号，可以通过设置环境变量 `export DEVICE_ID=x` 或者在context中设置 `device_id=x`指定相应的卡号。

运行时需配置config.sh
EPOCH_SIZE        # 训练的EPOCH_SIZE默认为90
RANK_TABLE_FILE   # 分布式训练需要，单机需注释掉
CONFIG_FILE       # 参数配置文件 例如resnet50_imagenet2012_Boost_config.yaml

