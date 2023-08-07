### 迁移预训练权重下载
[迁移预训练权重下载](https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz)

### 数据集下载

[测试数据集下载](https://www.image-net.org/challenges/LSVRC/2012/)

### 天数智芯 Iluvatar GPU配置与运行信息参考
#### 环境配置

- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本:  4.15.0-156-generic x86_64    
   - 加速卡驱动版本：3.1.0
   - Docker 版本：20.10.8
   - 训练框架版本：torch-1.13.1+corex.3.1.0
   - 依赖软件版本：无

### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(top1) | 性能（samples/s） |
| -------- | ------------- | ----------- | ------- | ------------ | ----------------- |
| 单机8卡  | config_BI-V100x1x8 |  | 0.83  |  0.8382   |       |
