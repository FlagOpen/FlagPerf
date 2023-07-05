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



* 通用指标

| 指标名称              | 指标值                 | 特殊说明                         |
| --------------------- | ---------------------- | -------------------------------- |
| 任务类别              | 图像分类               |                                  |
| 模型                  | faster_rcnn            | backbone=resnet50                |
| 数据集                | coco2017               |                                  |
| 数据精度              | precision,见“性能指标” | 可选fp32/amp/fp16                |
| 单卡批尺寸            | bs,见“性能指标”        | 即local batch_size               |
| 硬件设备简称          | nvidia A100            |                                  |
| 硬件存储使用(GiB)     | mem,见“性能指标”       | 通常称为“显存”                   |
| 端到端时间            | t1,见“性能指标”        | 时间单位为秒，性能单位为图片每秒 |
| 训练部分总时间/性能   | t2,见“性能指标”        | 不包含perf初始化、配置等部分     |
| 去除评估部分时间/性能 | t3,见“性能指标”        | 不包含每个epoch末尾的评估部分    |
| 计算部分时间/性能     | t4,见“性能指标”        | 不包含数据IO部分                 |
| 训练结果              | acc,见“性能指标”       | 单位为top1分类准确率(acc1)       |
| 额外修改项            | 无                     |                                  |

* 性能指标

| 配置               | precision | bs   | t1   | t2   | t3   | t4 | acc  | mem |
| ------------------ | --------- | ---- | ---- | ---- | ---- | ---- |  ---- | ---- |
| A100单机8卡（1x8） | fp32 | 2 |      |      |      |      |      | 8.1/40.0 |

