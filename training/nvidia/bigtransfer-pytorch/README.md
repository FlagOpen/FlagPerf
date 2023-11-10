### 迁移预训练权重下载
[迁移预训练权重下载](https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz)

### 数据集下载

[测试数据集下载](https://www.image-net.org/challenges/LSVRC/2012/)

### Nvidia GPU配置与运行信息参考
#### 环境配置

- ##### 硬件环境
    - 机器型号: NVIDIA DGX A100(40G) 
    - 加速卡型号: NVIDIA_A100-SXM4-40GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic     
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-1.8.0a0+52ea372
   - 依赖软件版本：无


### 运行情况



* 通用指标

| 指标名称       | 指标值                                          | 特殊说明                                    |
| -------------- | ----------------------------------------------- | ------------------------------------------- |
| 任务类别       | Image Classification && Representation Learning |                                             |
| 模型           | Big Transfer                                    |                                             |
| 数据集         | Imagenet2012 1K                                 |                                             |
| 数据精度       | precision,见“性能指标”                          | 可选fp32/amp/fp16/tf32                      |
| 超参修改       | fix_hp,见“性能指标”                             | 跑满硬件设备评测吞吐量所需特殊超参          |
| 硬件设备简称   | nvidia A100                                     |                                             |
| 硬件存储使用   | mem,见“性能指标”                                | 通常称为“显存”,单位为GiB                    |
| 端到端时间     | e2e_time,见“性能指标”                           | 总时间+Perf初始化等时间                     |
| 总吞吐量       | p_whole,见“性能指标”                            | 实际训练样本数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”                            | 不包含每个epoch末尾的评估部分耗时           |
| **计算吞吐量** | **p_core,见“性能指标”**                         | 不包含数据IO部分的耗时(p3>p2>p1)            |
| 训练结果       | acc,见“性能指标”                                | 单位为top1分类准确率(acc1)                  |
| 额外修改项     | 无                                              |                                             |

* 性能指标

| 配置              | precision | fix_hp | e2e_time | p_whole | p_train | p_core | final_acc1 | mem       |
| ----------------- | --------- | ------ | -------- | ------- | ------- | ------ | ---------- | --------- |
| A100单机8卡(1x8)  | fp32      | /      | 5869     | 222     | 225     | 228    | 0.84192    | 31.4/40.0 |
| A100单机8卡(1x8)  | fp32      | bs=20  | 5505     | 236     | 240     | 243    | 0.84016    | 37.4/40.0 |
| A100单机单卡(1x1) | fp32      | bs=16  | /     | 23.1    | 29.5    | 29.8   | /          | 38.1/40.0 |
| A100两机8卡(2x8)  | fp32      | bs=20  | /     | 459     | 465     | 470    | /          | 36.6/40.0 |


训练精度来源：https://paperswithcode.com/paper/large-scale-learning-of-general-visual

训练精度未对齐(84.11 VS 85.39)原因：没有采用x4迁移权重。后者在40GB显卡上，不使用FSDP优化无法训练batchsize=16，显存不足。
