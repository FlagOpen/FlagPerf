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
| 模型           | llama1_7B                    |                                             |
| 数据集         | openwebtext              |                                             |
| 配置文件       | config                    |                                             |
| 数据精度       | precision,见“性能指标”         | 可选fp32/amp/fp16                           |
| 超参修改       | fix_hp,见“性能指标”            | 跑满硬件设备评测吞吐量所需特殊超参          |
| 并行策略       | parallel_strategy,见“性能指标” | DP, TP, PP, SP          |
| 硬件设备简称   | nvidia A100 (80G *8)          |                                             |
| 硬件存储使用   | memory(actual/total),见“性能指标” | 通常称为“显存”,单位为GiB                    |
| 吞吐量       | throughput,见“性能指标”           | 训练吞吐量 |

* 性能指标

| 配置     | config | precision | fix_hp | parallel_strategy | throughput   | memory  |
| ------- | ------- | --------- | ------ | ---------------- | ------------ | ------ | 
| A100单机8卡（1x8）  | config_TP1PP1SH2SP8A100x1x8 | fp16, level="O2" | per_device_bs=4, accumulate=32, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=8 |   15.70715 * 2048 / 8 = 4021 tokens/s   |  76.98 * 8 GB  |
| A100单机8卡（1x8）  | config_TP2PP1SH1SP4A100x1x8 | fp16, level="O2" | per_device_bs=4, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |   14.27326 * 2048 / 8 = 3653 tokens/s   |  62.11 * 8 GB  |   
| A100单机8卡（1x8）  | config_TP2PP1SH2SP4A100x1x8 | fp16, level="O2" | per_device_bs=4, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |   13.48227 * 2048 / 8 = 3451 tokens/s   |  57.63 * 8 GB  |   
| A100单机8卡（1x8）  | config_TP2PP4SH1SP1A100x1x8 | fp16, level="O2" | per_device_bs=4, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |  13.644565 * 2048 / 8 = 3493 tokens/s   |  58.62\*2 + 53.51\*2 + 49.46\*2 + 47.95\*2 GB  |   