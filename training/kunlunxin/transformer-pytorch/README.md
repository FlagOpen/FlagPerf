### 测试数据集下载

[测试数据集下载](../../benchmarks/transformer/README.md#数据集)

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
  - Docker镜像和版本：xmlir/xmlir_ubuntu_2004_x86_64:v0.24
  - 训练框架版本：xmlir+a28ac56f
  - 依赖软件版本：pytorch-1.12.1+cpu


### 运行情况

| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能（tokens/s) |
| -------- | --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机8卡  | config_R300x1x8 |           |   27.0   |   27.27  |   24370 |                |

[官方精度](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#training-performance-nvidia-dgx-a100-8x-a100-40gb)为27.92，按照[官方配置](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#training-performance-nvidia-dgx-a100-8x-a100-40gb)，训完得到的精度为27.08
