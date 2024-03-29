### 测试数据集下载
[测试数据集下载](../../benchmarks/mobilenetv2/README.md#数据集)

### 昆仑芯XPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
  - 机器型号: 昆仑芯AI加速器组R480-X8
  - 加速卡型号: 昆仑芯AI加速卡R300
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：4.0.25
  - Docker镜像和版本：pytorch1.12.1-cpu-ubuntu18.04:v0.04
  - 训练框架版本：xmlir+e70db8f6
  - 依赖软件版本：pytorch-1.12.1+cpu


### 运行情况
* 通用指标

| 指标名称       | 指标值                  | 特殊说明                                    |
| -------------- | ----------------------- | ------------------------------------------- |
| 任务类别       | 图像分类                |                                             |
| 模型           | MobilenetV2             |                                             |
| 数据集         | ImageNet2012            |                                             |
| 数据精度       | precision,见“性能指标”  | 可选fp32/amp/fp16                           |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参          |
| 硬件设备简称   | R300                    |                                             |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB                    |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间                     |
| 总吞吐量       | p_whole,见“性能指标”    | 实际训练图片数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时           |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1)            |
| 训练结果       | acc,见“性能指标”        | 单位为top1分类准确率(acc1)                  |
| 额外修改项     | 无                      |                                             |



* 性能指标

| 配置                | precision | fix_hp         | e2e_time | p_whole | p_train | p_core | acc    | mem       |
| ------------------- | --------- | -------------- | -------- | ------- | ------- | ------ | ------ | --------- |
| R300单机单卡（1x1） | fp32      | /              | /        |         |         |        | /      | 24.8/32.0 |
| R300单机8卡（1x8）  | fp32      | bs=256,lr=0.36 |          |         |         |        | 68.43% | 24.8/32.0 |
| R300两机8卡（2x8）  | fp32      | bs=256,lr=0.72 | /        |         |         |        | /      | 26.7/32.0 |
### 许可证

Apache 2.0 license。
