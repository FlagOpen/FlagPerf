### 测试数据集下载

[测试数据集下载](../../benchmarks/transformer/README.md#数据集)

### Nvidia GPU配置与运行信息参考

#### 环境配置

- ##### 硬件环境

  - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境

  - OS版本：Ubuntu 18.04
  - OS kernel版本: 5.4.0-113-generic
  - 加速卡驱动版本：470.129.06
  - Docker 版本：20.10.16
  - 训练框架版本：1.12.1+cu113
  - 依赖软件版本：无


### 运行情况

| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能（tokens/s) |
| -------- | --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机1卡  | config_A100x1x1 |             |          |          |         |                  |
| 单机2卡  | config_A100x1x2 |             |          |          |         |                  |
| 单机4卡  | config_A100x1x4 |             |          |          |         |                  |
| 单机8卡  | config_A100x1x8 |    3.6h     |   3.90   |   3.87   |   58290 |     329423       |
| 两机8卡  | config_A100x2x8 |             |          |          |         |                  |

[官方精度](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#training-performance-nvidia-dgx-a100-8x-a100-40gb)为6.81，按照[官方配置](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#training-performance-nvidia-dgx-a100-8x-a100-40gb)，训完得到的精度为7.244，后续排期优化
