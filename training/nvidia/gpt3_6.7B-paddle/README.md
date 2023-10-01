
# Paddle版本运行指南

## 数据下载

```shell
mkdir GPT-3-data # 后续在training/run_benchmarks/config/test_conf.py中修改数据位置使用
wget -O GPT-3-data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O GPT-3-data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

## 基于FlagPerf运行

```
export case_name=gpt-6.7b # or gpt-13b
export nnodes=1 # 机器数量
export nprocs=8 # 机器可用GPU数量
export hosts="127.0.0.1" # 可用机器IP地址 “10.1.2.1, 10.1.2.3, 10.1.2.4”
export master="127.0.0.1"
export data_dir="./GPT-3-data"
export log_dir="./log_dir/results"


# 
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
    --extern_config_file config_A100x${nnodes}x${nprocs}.py \
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

    TODO: 需要进一步确认系统配置

    - 机器型号: NVIDIA DGX A100(40G) 
    - 加速卡型号: NVIDIA_A100-SXM4-40GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本: paddle-2.5.1
   - 依赖软件版本：
     - cuda: cuda_11.2.r11.2


### 运行情况

* GPT-3 (6.7B)
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s)|
| -------- | --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机1卡  | config_A100x1x1 | N/A         | 0.67     | N/A      | N/A     | N/A              |
| 单机2卡  | config_A100x1x2 | N/A         | 0.67     | N/A      | N/A     | N/A              |
| 单机4卡  | config_A100x1x4 | 1715.28     | 0.67     | 0.6809   | 6250    | 180.07           |
| 单机8卡  | config_A100x1x8 | 1315.42     | 0.67     | 0.6818   | 4689    | 355.63           |
| 双机2*8卡  | config_A100x2x8 | 1315.42     | 0.67     | 0.6818   | 4689    | 355.63           |

TODO: 不同配置的不同数据需要进一步测试后添加。

* GPT-3 (13B)
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s)|
| -------- | --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机1卡  | config_A100x1x1 | N/A         | 0.67     | N/A      | N/A     | N/A              |
| 单机2卡  | config_A100x1x2 | N/A         | 0.67     | N/A      | N/A     | N/A              |
| 单机4卡  | config_A100x1x4 | 1715.28     | 0.67     | 0.6809   | 6250    | 180.07           |
| 单机8卡  | config_A100x1x8 | 1315.42     | 0.67     | 0.6818   | 4689    | 355.63           |
| 双机2*8卡  | config_A100x2x8 | 1315.42     | 0.67     | 0.6818   | 4689    | 355.63           |

TODO: 不同配置的不同数据需要进一步测试后添加。