### 数据集下载

[数据集下载](https://www.image-net.org/challenges/LSVRC/2012/)

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
   - 训练框架版本：pytorch-1.8.0a0+52ea372
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标train_loss | 收敛train_loss | 性能(samples/s) |
| -------- | --------------- | ----------- | -------------- | -------------- | --------------- |
| 单机8卡  | config_A100x1x8 | 21596.08    | 0.35           | 0.3477         | 531.79          |

训练精度来源：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#results




### 许可证

本项目基于Apache 2.0 license。

本项目部分代码基于NVIDIA开源库 https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2 实现。