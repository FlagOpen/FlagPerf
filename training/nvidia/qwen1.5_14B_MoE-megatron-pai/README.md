### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### A800硬件环境
    - 机器型号: NVIDIA DGX A800(80G) 
    - 加速卡型号: NVIDIA_A800-SXM4-80GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### A800软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-126-generic     
   - 加速卡驱动版本：470.141.10
   - Docker 版本：20.10.18
   - 训练框架版本：Megatron-LM-240405

- ##### 并行策略

   - 并行技术：Tensor parallelism
   - 实施者：Pai-Megatron-Patch

- ##### 优化策略

   - flash attention 2

### 运行情况

* 输入批尺寸

1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A800x1x8.py中所写，在本case中默认为1
2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A800x1x8.py中所写，在本case中默认为128
3. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size。在本case中默认为8

* 通用指标

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 自然语言理解               |                                    |
| 模型         | qwen1.5_14B_MoE                  |                                    |
| 数据集       | pile wikipedia   |                                    |
| 数据精度       | precision,见“性能指标”  | 可选fp16/bf16                      |
| 硬件设备简称 | nvidia A800                |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |

* 性能指标

本例训练10000 step，此项实验也将作为精度对齐所用实验。精度对齐需第2001step及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。

| 配置                |  precision | parallel |  fix_hp           | token/p/s | lm loss value| lm loss PPL|mem       | MFU       |
| ------------------ | -------- | --------- | ---------------- | ------ | --------| ------- | --------- | --------- |
| A800单机8卡（4x8）  |    bf16    | TP1PP1DP1 |  / | 210.45 | 4.82 | 124.5 |66/80 | 28.3% |
| A800单机8卡（4x8）  |     fp16    | TP1PP1DP1 |  /| 257.60 | 4.82 | 124.5 |62/80 | 34.6% |
