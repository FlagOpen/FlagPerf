### 模型Checkpoint下载
* 运行自动下载


### 测试数据集下载
测试数据集中提供了处理好的100k条doc的训练样本：
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
```

### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器型号: NVIDIA DGX A100(80G) 
    - 加速卡型号: NVIDIA_A100-SXM4-80GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic     
   -  加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：paddle-2.5.1
   - 依赖软件版本：无

#### 运行情况

* 通用指标

| 指标名称       | 指标值                         | 特殊说明                                    |
| -------------- | ------------------------------ | ------------------------------------------- |
| 任务类别       | 文本分类、文本生成             |                                             |
| 模型           | llama1                    |                                             |
| 数据集         | openwebtext              |                                             |
| 配置文件       | config                    |                                             |
| 数据精度       | precision,见“性能指标”         | 可选fp32/amp/fp16                           |
| 超参修改       | fix_hp,见“性能指标”            | 跑满硬件设备评测吞吐量所需特殊超参          |
| 并行策略       | parallel_strategy,见“性能指标” | DP, TP, PP, SP          |
| 硬件设备简称   | nvidia A100 (80G * 8) and (40G * 8) |                                             |
| 硬件存储使用   | memory(actual/total),见“性能指标” | 通常称为“显存”,单位为GiB                    |
| 吞吐量       | throughput,见“性能指标”           | 训练吞吐量 |

* 性能指标

| 配置     | config | precision | fix_hp | parallel_strategy | throughput   | memory  |
| ------- | ------- | --------- | ------ | ---------------- | ------------ | ------ |
| LLaMA-7B | ------- | --------- | ------ | ---------------- | ------------ | ------ | 
| A100单机8卡（1x8*80G）  | config_TP1PP1SH2SP8A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=64, (global bs = 4M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=8 |   16.67 * 2048 / 8 = 4267 tokens/s   |  70.09 * 8 GB  |
| A100单机8卡（1x8*80G）  | config_TP2PP1SH1SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=128, (global bs = 4M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |   15.19 * 2048 / 8 = 3888 tokens/s   |  58.73 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP1SH2SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=128, (global bs = 4M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |   14.26 * 2048 / 8 = 3650 tokens/s   |  54.01 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP4SH1SP1A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=512, (global bs = 4M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage1", tensor_parallel_degree=2, pipline_parallel_degree=4 |  14.54 * 2048 / 8 = 3722 tokens/s   |  46.80\*2 + 38.93\*2 + 31.74\*2 + 26.92\*2 GB  |   
| LLaMA-7B | ------- | --------- | ------ | ---------------- | ------------ | ------ | 
| A100单机8卡（1x8*40G）  | config_TP1PP1SH2SP8A10040Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=128, (global bs =4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=False, sharding="stage2", sharding_degree=8 |   10.72 * 2048 / 8 = 2744 tokens/s   |  33.55 * 8 GB  |
| A100单机8卡（1x8*40G）  | config_TP2PP1SH1SP4A10040Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=256, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=False, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |   8.45 * 2048 / 8 = 2163 tokens/s   |  28.4 * 8 GB  |   
| A100单机8卡（1x8*40G）  | config_TP2PP1SH2SP4A10040Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=256, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=False, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |   8.44 * 2048 / 8 = 2160 tokens/s   |  25.8 * 8 GB  |   
| A100单机8卡（1x8*40G）  | config_TP2PP4SH1SP1A10040Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=1024, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=False, sharding="stage1", tensor_parallel_degree=2, pipline_parallel_degree=4 |  8.72 * 2048 / 8 = 2232 tokens/s   |  20.41\*2 + 19.80\*2 + 19.41\*2 + 20.12\*2 GB  |  
| LLaMA-13B | ------- | --------- | ------ | ---------------- | ------------ | ------ | 
| A100单机8卡（1x8*80G）  | config_TP1PP1SH2SP8A10080Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=128, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage2", sharding_degree=8 |   6.67 * 2048 / 8 = 1707 tokens/s   |  60.06 * 8 GB  |
| A100单机8卡（1x8*80G）  | config_TP2PP1SH1SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=256, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |   6.27 * 2048 / 8 = 1605 tokens/s   |  52.27 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP1SH2SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=256, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |   5.82 * 2048 / 8 = 1489 tokens/s   |  43.84 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP4SH1SP1A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=512, (global bs = 4M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage1", tensor_parallel_degree=2, pipline_parallel_degree=4 |  6.24 * 2048 / 8 = 1597 tokens/s   |  57.67\*2 + 46.49\*2 + 35.24\*2 + 25.59\*2 GB  |  