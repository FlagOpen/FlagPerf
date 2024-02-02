### 模型backbone权重下载
[模型backbone权重下载](https://download.pytorch.org/models/resnet50-0676ba61.pth)

这一部分路径在FlagPerf/training/benchmarks/retinanet/pytorch/model/\_\_init__.py中提供：

```python
torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1.value.url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
```
本case中默认配置为，从官网同路径（0676ba61）自动下载backbone权重。用户如需手动指定，可自行下载至被挂载到容器内的路径下，并于此处修改路径为"file://"+download_path

### 测试数据集下载

[测试数据集下载](https://cocodataset.org/)

### 沐曦集成电路 C500 GPU配置与运行信息参考
#### 环境配置

- ##### 硬件环境
    - 机器、加速卡型号: 曦云®C500 64G
    - 多机网络类型、带宽: InfiniBand，2x200 Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04.6
   - OS kernel版本:  5.4.0-26-generic
   - 加速卡驱动版本：2.2.0
   - Docker 版本：24.0.7
   - 训练框架版本：pytorch-2.0.0+mc2.18.0.8-cp38-cp38-linux_x86_64.whl
   - 依赖软件版本：无


* 通用指标

| 指标名称       | 指标值                  | 特殊说明                                       |
| -------------- | ----------------------- | ---------------------------------------------- |
| 任务类别       | 目标检测                |                                                |
| 模型           | retinanet               |                                                |
| 数据集         | COCO2017                |                                                |
| 数据精度       | precision,见“性能指标”  | 可选fp32/amp/fp16                              |
| 单卡批尺寸     | bs,见“性能指标”         | 即local batch_size                             |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参             |
| 硬件设备简称   | MXC500                 |                                                |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB                       |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间                        |
| 总吞吐量       | p_whole,见“性能指标”    | 实际训练图片数除以总时间(performance_whole)    |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时              |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1)               |
| 训练结果       | mAP,见“性能指标”        | 所有类别的 Average Precision（平均精度）的均值 |
| 额外修改项     | 无                      |                                                |

* 性能指标

| 配置                | precision | fix_hp        | e2e_time | p_whole | p_train | p_core | mAP    | mem       |
| ------------------- | --------- | ------------- | -------- | ------- | ------- | ------ | ------ | --------- |
| MXC500 单机8卡（1x8）  | fp32    | /             |         |          |        |        | 0.3485 | 37.5/64.0 |
| MXC500 单机单卡（1x1） | fp32    | /             | /       |          |        |        | /      | 25.1/64.0 |
| MXC500 两机8卡（2x8）  | fp32    | /             | /       |          |        |        | /      | 59.6/64.0 |

