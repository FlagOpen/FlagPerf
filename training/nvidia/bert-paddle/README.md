
### 模型Checkpoint下载
[模型Checkpoint下载](../../benchmarks/bert/README.md#模型checkpoint下载)


### 测试数据集下载
[测试数据集下载](../../benchmarks/bert/README.md#测试数据集下载)


### Paddle版本运行指南

单卡运行命令：
● 依赖包，paddlepaddle-gpu

'''
python -m pip install paddlepaddle-gpu==2.4.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple
'''

● bash环境变量:
```
export MASTER_ADDR=user_ip
export MASTER_PORT=user_port
export WORLD_SIZE=1
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1#可用的GPU索引
export RANK=0
export LOCAL_RANK=0
```
example：
```
export MASTER_ADDR=10.21.226.184
export MASTER_PORT=29501
export WORLD_SIZE=1
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1#可用的GPU索引
export RANK=0
export LOCAL_RANK=0
```

● 运行脚本:

在该路径目录下

```
python run_pretraining.py
--data_dir data_path
--extern_config_dir config_path
--extern_config_file config_file.py
```

example：
```
python run_pretraining.py
--data_dir /ssd2/yangjie40/data_config
--extern_config_dir /ssd2/yangjie40/flagperf/training/nvidia/bert-pytorch/config
--extern_config_file config_A100x1x2.py
```


### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: InfiniBand，200Gb/s
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本: paddle-2.4.0-rc
   - 依赖软件版本：
     - cuda: cuda_11.2.r11.2


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s)|
| -------- | --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机1卡  | config_A100x1x1 | N/A         | 0.67     | N/A      | N/A     | N/A              |
| 单机2卡  | config_A100x1x2 | N/A         | 0.67     | N/A      | N/A     | N/A              |
| 单机4卡  | config_A100x1x4 | 1715.28     | 0.67     | 0.6809   | 6250    | 180.07           |
| 单机8卡  | config_A100x1x8 | 1315.42     | 0.67     | 0.6818   | 4689    | 355.63           |

