### 数据集下载

[数据集下载](../../benchmarks/tacotron2/README.md#数据集下载地址)

### 天数智芯 BI-V100 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本:  4.15.0-156-generic x86_64    
   - 加速卡驱动版本：3.0.0
   - Docker 版本：20.10.8
   - 训练框架版本：torch-1.10.2+corex.3.0.0
   - 依赖软件版本：无



### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标val_loss | 收敛val_loss | 性能(samples/s) |
| -------- | --------------- | ----------- | ------------ | ------------ | --------------- |
| 单机8卡  | config_BI-V100x1x8 |   38760.58  |  0.4852     |    0.4833    |     71.98      |

注：
训练精度来源：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#results，根据官方仓库中的脚本，训练1500epoch得到val_loss=0.4852.
