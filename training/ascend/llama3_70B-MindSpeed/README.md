
### Ascend 配置与运行信息参考
#### 环境配置
- ##### Atlas 800T A3硬件环境
    
    - 加速卡型号: /
    - 多机网络类型、带宽: /

   - 训练框架版本：megatron-core tag:core_v0.8.0

- ##### 并行策略

   - 并行技术：张量、流水、数据混合并行，具体并行方案见“运行情况”章节
   - 实施者：megatron-core
   - 实施细节：PP4TP8VPP2

- ##### 优化策略

   - flash attention 2
   - reuse-fp32-param
   - reset-position-ids
   - use-distributed-optimizer
   - overlap-grad-reduce
   - overlap-param-gather
   - use-rotary-position-embeddings
   - use-fused-rotary-pos-emb
   - use-fused-rmsnorm
   - use-fused-swiglu

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为1
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为8192
  3. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size。为256

* 通用指标

| 指标名称    | 指标值                   | 特殊说明                                     |
| ------- | --------------------- | ---------------------------------------- |
| 任务类别    | 自然语言理解                |                                          |
| 模型      | llama3_70b             |                                          |
| 数据集     | enwiki                 |                                          |
| 数据精度    | precision,见“性能指标”     | 可选fp32/amp/fp16/bf16                     |
| 超参修改    | parallel,见“性能指标”      | 格式为PPxDPyTPz，例如PP2DP2TP8                 |
| 超参修改    | fix_hp,见“性能指标”        | 跑满硬件设备评测吞吐量所需特殊超参                        |
| 硬件设备简称  |  A3                |                                          |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB                          |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                               |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数                          |

* 性能指标


| 配置                 | precision | parallel    | fix_hp | token/p/s | 是否精度对齐     | mem   | MFU         |
| ------------------ | --------- | ----------- | ------ | --------- | ---------- | ----- | ----------- |
| A3四机32卡（4x8）     | /      | /   | /      |       | /       | / | / |

