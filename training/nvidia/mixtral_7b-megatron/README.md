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
   - 训练框架版本：Megatron-LM

- ##### 并行策略

   - 并行技术：Tensor parallelism
   - 实施者：Megatron-LM


### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A800x1x8.py中所写，在本case中默认为1
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A800x1x8.py中所写，在本case中默认为2048。
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为config_A800x1x8中所写，在本case中默认为1
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size，简写为GBS。


* 通用指标

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 多专家混合模型               |                                    |
| 模型         | mixtral_7x8B                  |                                    |
| 数据集       |                 | |
| 数据精度     |bf16                        |                                    |
| 超参修改     | fix_hp,见“性能指标”        | 运行必要特殊超参 |
| 硬件设备简称 | nvidia A800                |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |
* 性能指标

| 配置                |  fix_hp           | token/p/s | loss | mem       |MFU       |
| ------------------- | ---------------- | ------ | ------- | --------- |--------- |
| A800单机8卡（2x8） |  /  | - | - | - | / |-|
