### 迁移预训练权重下载
[迁移预训练权重下载](https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz)

### 数据集下载

[测试数据集下载](https://www.image-net.org/challenges/LSVRC/2012/)

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
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(top1) | 性能（samples/s） |
| -------- | --------------- | ----------- | -------- | ------------- | ----------------- |
| 单机8卡  | config_A100x1x8 | 5771.27 | 0.83  | 0.8411     | 222.02       |

训练精度来源：https://paperswithcode.com/paper/large-scale-learning-of-general-visual

训练精度未对齐(84.11 VS 85.39)原因：没有采用x4迁移权重。后者在40GB显卡上，不使用FSDP优化无法训练batchsize=16，显存不足。
