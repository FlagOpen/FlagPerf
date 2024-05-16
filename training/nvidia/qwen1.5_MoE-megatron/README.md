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

   - 并行技术：Pipeline parallel
   - 实施者：Pai-Megatron-Patch

- ##### 优化策略

   - flash attention 2

- ##### 依赖环境

   - sentencepiece==0.2.0
   - 注：不同版本的sentencepiece分词方式可能会有所不同，为了训练误差的比对，在本case中将sentencepiece的版本设置为0.2.0

### 运行情况

* 输入批尺寸

1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A800x1x8.py中所写，在本case中默认为1。
2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A800x1x8.py中所写，在本case中默认为8192。
3. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size。在本case中默认为512。
4. 在本case中，data_parallel_size=world_size/TPsize/PPsize。

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

本例训练100 step，此项实验也将作为精度对齐所用实验。精度对齐需第21step及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。

`注：因原仓库没有提供精度参考，因此我们基于源代码跑出NV版本的loss作为参考值。实验显示lm loss value随机初始值约为12，训练100轮后降到了8.53，呈现收敛特性。将其设为参考值是为了比较其他芯片在相同的配置下，loss的降低过程是否匹配，以进一步定量对比 mem 与 MFU 。`

| 配置                |  precision | parallel |  fix_hp           | token/p/s | lm loss value| mem       | MFU       |
| ------------------ | -------- | --------- | ---------------- | ------ |  ------- | --------- | --------- |
| A800单机8卡（1x8）  |    bf16    | TP1PP2DP4 |  / | 7450.7 | 8.53 | 66/80 | 32.95% |
| A800单机8卡（1x8）  |    fp16    | TP1PP2DP4 |  /| 7537.7 | 8.59 | 59/80 | 33.33% |