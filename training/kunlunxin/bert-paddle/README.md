
### 模型Checkpoint下载
[模型Checkpoint下载](../../benchmarks/bert/README.md#模型checkpoint下载)


### 测试数据集下载
[测试数据集下载](../../benchmarks/bert/README.md#测试数据集下载)


### Paddle版本运行指南

● bash环境变量:
```
export FLAGS_sync_nccl_allreduce=0
export FLAGS_fraction_of_gpu_memory_to_use=0.99
export FLAGS_call_stack_level=2
export FLAGS_use_fast_math=0
export FLAGS_enable_nvtx=1
export BKCL_CCIX_RING=1
export XPU_PADDLE_L3_SIZE=41943040
export XPU_PADDLE_FC_TRANS_A=1
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  #可用xpu索引
```

● 运行脚本:

在该路径目录下

```
python -u  -m paddle.distributed.launch --xpus=${XPU_VISIBLE_DEVICES} run_pretraining.py \
--data_dir data_path \
--extern_config_dir config_path \
--extern_config_file config_file.py

```


example：
```
python -u  -m paddle.distributed.launch --xpus=${XPU_VISIBLE_DEVICES} run_pretraining.py \
--data_dir /bert-data/train \
--extern_config_dir /home/FlagPerf/training/kunlunxin/bert-paddle/config \
--extern_config_file config_R300x1x8.py
```


### 昆仑芯XPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
  - 机器型号: 昆仑芯AI加速器组R480-X8
  - 加速卡型号: 昆仑芯AI加速卡R300
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：4.0.25
  - Docker镜像和版本：registry.baidubce.com/paddlepaddle/paddle:2.3.2
  - 训练框架版本：paddlepaddle+f6161d1
  - 依赖软件版本：pytorch-1.8.1
  - 



### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s)|
| -------- | --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机8卡  | config_A100x1x8 |       | 0.67     | 0.6709   | 11720    |              |

### 许可证

本项目基于Apache 2.0 license。

本项目部分代码基于MLCommons https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/ 实现。
