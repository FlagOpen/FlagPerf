### 数据集下载

[数据集下载](../../benchmarks/tacotron2/README.md#数据集下载地址)

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

| 指标名称       | 指标值                  | 特殊说明                                    |
| -------------- | ----------------------- | ------------------------------------------- |
| 任务类别       | SpeechSynthesis         |                                             |
| 模型           | tacotron2               |                                             |
| 数据集         | LJSpeech                |                                             |
| 数据精度       | precision,见“性能指标”  | 可选fp32/amp/fp16/tf32                      |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参          |
| 硬件设备简称   | nvidia A100             |                                             |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB                    |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间                     |
| 总吞吐量       | p_whole,见“性能指标”    | 实际训练样本数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时           |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1)            |
| 训练结果       | val_loss,见“性能指标”   | 验证loss                                    |
| 额外修改项     | 无                      |                                             |

* 性能指标

| 配置              | precision | fix_hp          | e2e_time | p_whole | p_train | p_core | val_loss | mem       |
| ----------------- | --------- | --------------- | -------- | ------- | ------- | ------ | -------- | --------- |
| A100单机8卡(1x8)  | tf32      | /               | 10719    | 257556  | 265661  | 280476 | 0.4774   | 37.5/40.0 |
| A100单机单卡(1x1) | tf32      | bs=128,lr=0.001 |    /      | 34440   | 34591   | 35562  |    /      | 35.2/40.0 |
| A100两机8卡(2x8)  | tf32      | bs=128,lr=0.001 |    /      | 484402  | 512004  | 558171 |    /      | 37.7/40.0 |



注：
训练精度来源：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#results，根据官方仓库中的脚本，训练1500epoch得到val_loss=0.4852.
tacotron2官网仓库的训练日志见PR附件。
