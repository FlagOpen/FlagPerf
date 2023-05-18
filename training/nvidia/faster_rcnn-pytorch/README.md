### 模型backbone权重下载
[模型backbone权重下载](https://download.pytorch.org/models/resnet50-0676ba61.pth)

这一部分路径在FlagPerf/training/benchmarks/faster_rcnn/pytorch/model/\_\_init__.py中提供：

```python
torchvision.models.resnet.__dict__['model_urls'][
    'resnet50'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
```
本case中默认配置为，从官网同路径（0676ba61）自动下载backbone权重。用户如需手动指定，可自行下载至被挂载到容器内的路径下，并于此处修改路径为"file://"+download_path

### 测试数据集下载

[测试数据集下载](https://cocodataset.org/)

### Nvidia GPU配置与运行信息参考
#### 环境配置

- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic     
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-1.8.0a0+52ea372
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(mAP) | 性能（samples/s） |
| -------- | --------------- | ----------- | -------- | ------------- | ----------------- |
| 单机1卡  | config_A100x1x1 |     |    |       |             |
| 单机4卡  | config_A100x1x4 |    |    |        |             |
| 单机8卡  | config_A100x1x8 | 16730.435    | 0.35     | 0.3520        | 182.24            |
| 两机8卡  | config_A100x2x8 |      |      |        |            |

训练精度来源：[torchvision.models — Torchvision 0.8.1 documentation (pytorch.org)](https://pytorch.org/vision/0.8/models.html?highlight=faster#torchvision.models.detection.fasterrcnn_resnet50_fpn)

### 许可证

本项目基于Apache 2.0 license。

本项目部分代码基于torchvision https://github.com/pytorch/vision/tree/release/0.9/references/detection 实现。
