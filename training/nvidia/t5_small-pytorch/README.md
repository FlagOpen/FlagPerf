### 1. 下载数据集和模型
[下载链接](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/datasets/t5_small_train.tar) 

### 2. 设置test_conf.py

为了使得`training/nvidia/t5_small-pytorch/config/requirements.txt`里的依赖库均能被下载，需要将`training/run_benchmarks/config/test_conf.py`里的`PIP_SOURCE`的值修改为`https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. Nvidia GPU配置与运行信息参考
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
   - 依赖软件版本：
     - cuda: 11.4

### 运行情况

* 通用指标

| 指标名称       | 指标值                  | 特殊说明                              |
| -------------- | ----------------------- | ------------------------------------- |
| 任务类别       | Summarization                |                                       |
| 模型           | t5_small                |                                       |
| 数据集         | CNN/Daily Mail            |                                       |
| 超参修改       | fix_hp,见“性能指标” | 跑满硬件设备评测吞吐量所需特殊超参 |
| 硬件设备简称   | nvidia A100             |                                       |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB              |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间               |
| 总吞吐量       | p_whole,见“性能指标”    | 实际样本数数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时     |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1)      |
| 训练结果       | rouge1,见“性能指标”        | rouge1分数            |
| 训练结果       | rouge2,见“性能指标”        | rouge2分数            |
| 训练结果       | rougeL,见“性能指标”        | rougeL分数            |
| 训练结果       | rougeLsum,见“性能指标”        | rougeLsum分数            |
| 额外修改项     | 无                      |                                       |

* 性能指标

| 配置               | precision | fix_hp | e2e_time | p_whole | p_train | p_core | rouge1  | rouge2  | rougeL  | rougeLsum  | mem |
| ------------------ | --------- | ---- | ---- | ---- | ---- | ---- |  ---- | ---- | ---- | ---- | ---- |
| A100单机8卡（1x1） | fp32 | / | | | | | | | | | |
| A100单机8卡（1x8） | fp32 | / | 996.11 | 338 | 398 | 400 | 41.12 | 18.84 | 29.15 | 38.32 | 35.3 /40.0 |
| A100单机8卡（2x8） | fp32 | / | | | | | | | | | |

注意: T5模型MFU数值较低, 为11.8%
1x8训练的MFU计算过程如下:
`MFU = 400.26068691305795 * 1024 * (60 * 10^6) * 6 / (156 * 1000^4) / 8 = 11.8%`

其中, 1024为seq_len, 60 millions为参数量, (156 * 1000^4)为A100 tf32算力

