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

| 指标名称       | 指标值                  | 特殊说明                              |
| -------------- | ----------------------- | ------------------------------------- |
| 任务类别       | 图像目标检测            |                                       |
| 模型           | fasterRCNN              |                                       |
| 数据集         | coco2017                |                                       |
| 数据精度       | precision,见“性能指标”  | 可选fp32/amp/fp16                     |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参          |
| 硬件设备简称   | nvidia A100             |                                       |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB              |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间               |
| 总吞吐量       | p_whole,见“性能指标”    | 实际训练图片数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时     |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1)      |
| 训练结果       | map,见“性能指标”        | 单位为平均目标检测正确率              |
| 额外修改项     | 无                      |                                       |

* 性能指标

| 配置               | precision | fix_hp | e2e_time | p_whole | p_train | p_core | map | mem |
| ------------------ | --------- | ---- | ---- | ---- | ---- | ---- |  ---- | ---- |
| A100单机8卡（1x8） | fp32 | / | 14086 | 144     | 150 | 199 | 37.0% | 7.9/40.0 |
| A100单机8卡（1x8） | fp32 | bs=16,lr=0.16 | 11848 | 187 | 199 | 298 | 36.6% | 38.8/40.0 |
| A100单机8卡（1x1） | fp32 | bs=16,lr=0.16 |  | 26 | 27    | 45 |  |33.7/40.0 |
| A100两机8卡（2x8） | fp32 | bs=16,lr=0.16 |  | 351 | 383    | 601 |  | 39.0/40.0 |
