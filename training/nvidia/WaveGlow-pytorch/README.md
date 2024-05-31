### 数据集下载

[数据集下载](../../benchmarks/WaveGlow/README.md#数据集下载地址)

### Nvidia GPU配置与运行信息参考
#### 环境配置

- ##### 硬件环境
    - 机器型号: NVIDIA DGX A100(40G) 
    - 加速卡型号: NVIDIA_A100-SXM4-40GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic     
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-1.13
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标val_loss | 收敛val_loss | 性能(samples/s) |
| -------- | --------------- | ----------- | ------------ | ------------ | --------------- |
| 单机8卡  | config_A100x1x8 |  12851   | -5.72     |    -5.72    |    997479       |

注：
训练精度来源：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#results，根据官方仓库中的脚本，训练250epoch得到val_loss=-5.7602