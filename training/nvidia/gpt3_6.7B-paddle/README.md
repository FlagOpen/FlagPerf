
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
export case_name=gpt3_6.7B # or gpt3-13B
export nnodes=1 # 机器数量
export nprocs=8 # 机器可用GPU数量
export hosts="127.0.0.1" # 可用机器IP地址 “10.1.2.1, 10.1.2.3, 10.1.2.4”
export master="127.0.0.1"
export data_dir="./GPT-3-data"
export log_dir="./log_dir/results"

cd ${workspaceFolder}/training/run_benchmarks/

python paddle/start_paddle_task.py \
    --vendor nvidia \
    --case_name ${case_name}:paddle_2.5.1:A100:${nnodes}:${nprocs}:1 \
    --model_name ${case_name} \
    --train_script run_pretraining.py \
    --nnodes ${nnodes} \
    --nproc ${nprocs} \
    --hosts ${hosts} \
    --hosts_ports 2222 \ 
    --data_dir ${data_dir} \
    --log_dir ${log_dir} \
    --log_level debug \
    --extern_config_file config_TP1PP1SH2SP8A10080Gx${nnodes}x${nprocs}.py \
    --enable_extern_config --master_port 29301 \
    --round 1 \
    --visible_dev_env CUDA_VISIBLE_DEVICES \
    --master_addr ${master} \
    --node_rank 0 \
    --host_addr ${hosts}
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
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本: paddle-2.5.1
   - 依赖软件版本：
     - cuda: cuda_11.8

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
| 硬件设备简称   | nvidia A100 (80G * 8) and (40G * 8) |                                             |
| 硬件存储使用   | memory(actual/total),见“性能指标” | 通常称为“显存”,单位为GiB                    |
| 吞吐量       | throughput,见“性能指标”           | 训练吞吐量 |

* 性能指标

| 配置     | config | precision | fix_hp | parallel_strategy | throughput   | memory  |
| ------- | ------- | --------- | ------ | ---------------- | ------------ | ------ |
| GPT3-6.7B | ------- | --------- | ------ | ---------------- | ------------ | ------ | 
| A100单机8卡（1x8*80G）  | config_TP1PP1SH2SP8A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=32, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=8 |   17.23 * 2048 / 8 = 4410 tokens/s   |  66.27 * 8 GB  |
| A100单机8卡（1x8*80G）  | config_TP2PP1SH1SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |   15.72 * 2048 / 8 = 4024 tokens/s   |  54.65 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP1SH2SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |   14.89 * 2048 / 8 = 3811 tokens/s   |  49.46 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP4SH1SP1A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=256, (global bs = 2M tokens) | flash_attention=True, recompute=False, use_fused_rms_norm=True, sharding="stage1", tensor_parallel_degree=2, pipline_parallel_degree=4 |  14.81 * 2048 / 8 = 3791 tokens/s   |  46.77\*2 + 36.59\*2 + 30.09\*2 + 27.78\*2 GB  |    
| GPT3-13B | ------- | --------- | ------ | ---------------- | ------------ | ------ | 
| A100单机8卡（1x8*80G）  | config_TP1PP1SH2SP8A10080Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=64, (global bs = 2M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage2", sharding_degree=8 |   6.97 * 2048 / 8 = 1784 tokens/s   |  76.29 * 8 GB  |
| A100单机8卡（1x8*80G）  | config_TP2PP1SH1SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=128, (global bs = 2M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage1", sharding_degree=4, tensor_parallel_degree=2 |   6.44 * 2048 / 8 = 1648 tokens/s   |  50.72 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP1SH2SP4A10080Gx1x8 | fp16, level="O2" | per_device_bs=2, accumulate=128, (global bs = 2M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage2", sharding_degree=4, tensor_parallel_degree=2 |   5.93 * 2048 / 8 = 1518 tokens/s   |  48.54 * 8 GB  |   
| A100单机8卡（1x8*80G）  | config_TP2PP4SH1SP1A10080Gx1x8 | fp16, level="O2" | per_device_bs=4, accumulate=256, (global bs = 2M tokens) | flash_attention=True, recompute=True, use_fused_rms_norm=True, sharding="stage1", tensor_parallel_degree=2, pipline_parallel_degree=4 |  6.42 * 2048 / 8 = 1643 tokens/s   |  35.84\*2 + 34.71\*2 + 32.80\*2 + 32.04\*2 GB  | 