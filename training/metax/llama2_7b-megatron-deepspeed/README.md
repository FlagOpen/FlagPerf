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

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 自然语言理解               |                                    |
| 模型         | llama2_7b                  |                                    |
| 数据集       | RedPajama-Data-1T-Sample   |                                    |
| 数据精度     | amp                        |                                    |
| 超参修改     | fix_hp,见“性能指标”        | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称 | MXC500                |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |

* 性能指标

| 配置                |  fix_hp           | token/p/s | loss | mem       | MFU       |
| ------------------- | ---------------- | ------ | ------- | --------- | --------- |
| C500单机8卡（1x8）  |  MPE=4096 LBS=1 GAS=32 TP=1 PP=1  | / | 3.83 | 62/64 | 51.8% |
