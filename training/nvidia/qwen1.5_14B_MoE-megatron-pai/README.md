### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### A800硬件环境
    - 机器型号: NVIDIA DGX A800(80G) 
    - 加速卡型号: NVIDIA_A800-SXM4-80GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### A800软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-126-generic     
   - 加速卡驱动版本：470.141.10
   - Docker 版本：20.10.18
   - 训练框架版本：Megatron-LM-240405

- ##### 并行策略

   - 并行技术：Tensor parallelism
   - 实施者：Pai-Megatron-Patch

- ##### 优化策略

   - flash attention 2