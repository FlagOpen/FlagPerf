
# Paddle版本运行指南

## 数据下载

```shell
mkdir GPT-3-data # 后续在training/run_benchmarks/config/test_conf.py中修改数据位置使用
cd GPT-3-data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

## 基于FlagPerf运行

```
cd FlagPerf/training
sudo -E python3 ./run_benchmarks/run.py
```


### GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: 曦云®C500 64G
    - CPU型号: Montage Jintide(R) C8458P
    - 多机网络类型、带宽: InfiniBand，2x200 Gb/s
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-26-generic
   - 加速卡驱动版本：2.2.0
   - Docker 版本：24.0.7
   - 训练框架版本: paddle-2.6.0
   - 依赖软件版本：sentencepiece

#### 运行情况

* 通用指标

| 指标名称       | 指标值                         | 特殊说明                                    |
| -------------- | ------------------------------ | ------------------------------------------- |
| 任务类别       | 文本分类、文本生成             |                                             |
| 模型           | gpt3                    |                                             |
| 数据集         | gpt_en_dataset              |                                             |
| 配置文件       | config                    |                                             |
| 数据精度       | precision,见“性能指标”         | 可选fp32/amp/fp16                           |
| 超参修改       | fix_hp,见“性能指标”            | 跑满硬件设备评测吞吐量所需特殊超参          |
| 并行策略       | parallel_strategy,见“性能指标” | DP, TP, PP, SP          |
| 硬件设备简称   | metax C500 (64G * 8) |                                             |
| 硬件存储使用   | memory(actual/total),见“性能指标” | 通常称为“显存”,单位为GiB                    |
| 吞吐量       | throughput,见“性能指标”           | 训练吞吐量 |

* 性能指标

| 配置     | config | precision | fix_hp | parallel_strategy | throughput   |
| ------- | ------- | --------- | ------ | ---------------- | ------------ |
| GPT3-6.7B | ------- | --------- | ------ | ---------------- | ------------ |
| C500单机8卡（1x8*64G） | config_TP1PP1SH2SP8C50040Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=true, use_fused_rms_norm=false, sharding="stage2", sharding_degree=8 |      |
| C500单机8卡（1x8*64G） | config_TP2PP1SH1SP4C50040Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=128, (global bs = 2M tokens) | flash_attention=True, recompute=true, use_fused_rms_norm=false, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |      |
|  |  |  |  |  |  |
