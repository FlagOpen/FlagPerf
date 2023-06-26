### 数据集下载

[数据集下载](../../benchmarks/compgcn/README.md#数据集下载地址)

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
   - 训练框架版本：pytorch-1.13.0a0+936e930
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标 MRR && Hit@1 | 收敛MRR && Hit@1 | 性能(samples/s) |
| -------- | --------------- | ----------- | ----------------- | ---------------- | --------------- |
| 单机8卡  | config_A100x1x8 | 15126.85    | 0.463 && 0.430    | 0.4639 && 0.4741 | 3448.26         |

注：
训练精度来源：https://github.com/dmlc/dgl/tree/master/examples/pytorch/compGCN#performance
根据官方仓库中的脚本，在WN18RR数据集上，训练500epoch得到MRR=0.466, Hit@1=0.435
