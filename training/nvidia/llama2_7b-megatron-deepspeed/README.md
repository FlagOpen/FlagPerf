### 测试数据集下载
[测试数据集下载](../../benchmarks/llama2_7b/megatron-deepspeed)

### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器型号: NVIDIA DGX A100(40G) 
    - 加速卡型号: NVIDIA_A100-SXM4-40GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s
    
- ##### H800软件环境
   - OS版本：Ubuntu 22.04 LTS
   - OS kernel版本: 5.15.0-25-generic     
   - 加速卡驱动版本：535.129.03
   - Docker 版本：24.0.7
   - 训练框架版本：deepspeed 0.11.0, Megatron-DeepSpeed.git@11f2d9342
   - 依赖软件版本：sentencepiece, transformers==4.34.1

- ##### 并行策略
    - 并行技术: data parallel, tensor parallel, pipeline parallel
    - 实施者：Megatron Deepspeed
- ##### 优化策略
    - flash attention 2

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A100x1x8.py中所写，在本case中默认为1
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A100x1x8.py中所写，在本case中默认为4096
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为ds_config.json中所写，在本case中默认为32
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size/(tensor_parallel\*pipeline_parallel)，简写为GBS
  5. tensor_parallel, 简写为TP, 本case默认为4
  6. pipeline_parallel, 简写为PP, 本case默认为1

* 通用指标

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 自然语言理解               |                                    |
| 模型         | llama2_7b                  |                                    |
| 数据集       | RedPajama-Data-1T-Sample   |                                    |
| 数据精度     | amp                        |                                    |
| 超参修改     | parallel,见“性能指标” | 格式为TPxPPyDPz，例如TP2PP1DP4            |
| 超参修改     | fix_hp,见“性能指标”        | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称 | nvidia A100                |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |

* 性能指标

值得注意的是，下列第4组实验的global_batchsize与llama2原始论文相同, 训练100 step，此项实验也将作为精度对齐所用实验。精度对齐需第21步及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。

| 配置                | parallel |  fix_hp           | token/p/s | loss | 是否精度对齐 | mem       | MFU       |
| ------------------- | -------- | ---------------- | ---------- | ---- | ---------- | ---------- | -------- |
| A100单机8卡（1x8）  | TP4PP1DP2 | /                | 2868       | 4.65 | /          | 32/40      | 38.6%    |
| A100单机8卡（1x8）  | TP4PP1DP2 | GAS=512(GBS=1024=4M tokens)  | 2908       | 7.86 | True          | 32/40      | 39.1%    |