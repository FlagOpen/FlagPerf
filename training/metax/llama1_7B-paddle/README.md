### 模型Checkpoint下载
* 运行自动下载


### 测试数据集下载
测试数据集中提供了处理好的100k条doc的训练样本：
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
```

### 沐曦集成电路 C500 GPU配置与运行信息参考 2.20.2.14
#### 环境配置
- ##### 硬件环境
    - 加速卡型号: 曦云®C500 64G
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200 Gb/s
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic     
   -  加速卡驱动版本：470.129.06
   - Docker 版本：24.0.7
   - 训练框架版本：paddle-2.6.0
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
| C500单机8卡（1x8*80G）  | config_TP1PP1SH2SP8C500x1x8 | fp16, level="O2" | per_device_bs=2, accumulate=128, (global bs = 4M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=False, sharding="stage2", sharding_degree=8 |    |  53.12 * 8 GB  |
