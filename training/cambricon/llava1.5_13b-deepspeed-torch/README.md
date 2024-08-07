### Cambricon MLU配置与运行信息参考
#### 环境配置
- ##### /硬件环境
    - 机器型号: /
    - 加速卡型号: /
    - CPU型号: /
    - 多机网络类型、带宽: /
    
- ##### /软件环境
   - OS版本：Ubuntu 22.04
   - OS kernel版本: 5.15.0-107-generic  
   - 加速卡驱动版本：v5.10.34
   - Docker 版本：25.0.3
   - 训练框架版本：deepspeed 0.13.2

- ##### 并行策略

   - 并行技术：sharded data parallel
   - 实施者：deepspeed ZeRO-DP
   - 实施细节：ZeRO-DP Pretrain-stage:O2 Finetune-stage:O3

- ##### 优化策略

   - flash attention 2

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_MLUx1x8.py中所写，在本case中pretrain阶段为32，finetune阶段为16
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_MLUx1x8.py中所写，在本case中默认为2048。这里需注意，llava1.5实际训练时，实际序列长度并非都为2048，本case在计算MFU时，统计每条数据进入模型的实际序列长度求取平均值作为实际序列长度
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为config_MLUx1x8中所写，在本case中默认为1
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size，简写为GBS。在本case中，只存在数据并行，因此data_parallel_size=world_size

- ##### 优化策略

   - 优化方案：flash attention 2


* 通用指标

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 多模态大模型               |                                    |
| 模型         | llava1.5_13b                  |                                    |
| 数据集       | LAION-CC-SBU、llava的混合指令微调数据                | |
| 数据精度     |bf16                        |                                    |
| 超参修改     | fix_hp,见“性能指标”        | 运行必要特殊超参 |
| 硬件设备简称 | cambricon /                |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |
| MMMU（val）结果           | acc(推理/验证)   | MMMU（val）回答准确率                   |
* 性能指标

| 配置                |  fix_hp           | token/p/s | loss | mem       |acc(MMMU) |MFU       |
| ------------------- | ---------------- | ------ | ------- | --------- | --------- |--------- |
| /单机8卡（1x8）（pretrain）  |  /  | / | / | / | / | / |
| /单机8卡（1x8）（finetune）  |  /  | / | / | / | / | / |
