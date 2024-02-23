### 测试数据集下载
[测试数据集下载](../../benchmarks/llama2_7b/megatron-deepspeed)

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
   - 训练框架版本：pytorch-2.0.0+mc2.18.0.8-cp38-cp38-linux_x86_64.whl, deepspeed 0.10.0, Megatron-DeepSpeed.git@11f2d9342
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
  5. tensor_parallel, 简写为TP, 本case默认为1
  6. pipeline_parallel, 简写为PP, 本case默认为1

* 通用指标

| 指标名称    | 指标值                      | 特殊说明                          |
| ------- | ------------------------ | ----------------------------- |
| 任务类别    | 自然语言理解                   |                               |
| 模型      | llama2_7b                |                               |
| 数据集     | RedPajama-Data-1T-Sample |                               |
| 数据精度    | amp                      |                               |
| 超参修改    | parallel,见“性能指标”         | 格式为TPxPPyDPz，例如TP2PP1DP4      |
| 超参修改    | fix_hp,见“性能指标”           | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称  | MXC500                   |                               |
| 硬件存储使用  | mem,见“性能指标”              | 通常称为“显存”,单位为GiB               |
| 计算使用率   | MFU,见“性能指标”              | 参见PaLM论文定义                    |
| **吞吐量** | **token/p/s,见“性能指标”**    | 平均单卡每秒处理的token数               |

* 性能指标

值得注意的是，下列第2组实验的global_batchsize与llama2原始论文相同, 训练100 step，此项实验也将作为精度对齐所用实验。精度对齐需第21步及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。

| 配置            | parallel  | fix_hp                      | token/p/s | 是否精度对齐 | mem   | MFU   |
| ------------- | --------- | --------------------------- | --------- | ------ | ----- | ----- |
| C500单机8卡（1x8） | TP1PP1DP8 | /                           | /         | /      | 62/64 | 51.8% |
| C500单机8卡（1x8） | TP4PP1DP2 | GAS=128(GBS=1024=4M tokens) | /         | True   | 62/64 | 53.0% |
