### Cambricon MLU配置与运行信息参考
#### 环境配置
- ##### MLU硬件环境
    - 机器型号: /
    - 加速卡型号: /
    - CPU型号: /
    - 多机网络类型、带宽: /

- ##### MLU软件环境
   - OS版本：Ubuntu 22.04 LTS
   - OS kernel版本: 5.15.0-107-generic     
   - 加速卡驱动版本：5.10.34
   - Docker 版本：24.04
   - 训练框架版本：deepspeed 0.10.1
   
- ##### 并行策略

   - 并行技术：sharded data parallel
   - 实施者：deepspeed ZeRO-DP
   - 实施细节：ZeRO-DP O1, DP_SIZE=8

- ##### 优化策略

   - gradient checkpointing

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_MLUx1x8.py中所写，在本case中默认为2
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_MLUx1x8.py中所写，在本case中默认为2048
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为ds_config.json中所写，在本case中默认为8，精度对齐实验默认为64
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size，简写为GBS。在本case中，只存在数据并行，因此data_parallel_size=world_size。

* 通用指标

| 指标名称    | 指标值                   | 特殊说明                          |
| ------- | --------------------- | ----------------------------- |
| 任务类别    | 自然语言理解                |                               |
| 模型      | baichuan2_13b         |                               |
| 数据集     | openwebtext           | 如无特殊说明，训练前1亿个token            |
| 数据精度    | bf16                  |                               |
| 超参修改    | fix_hp,见“性能指标”        | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称  | /           |                               |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB               |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                    |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数               |

* 性能指标

| 配置              | fix_hp                     | token/p/s | loss | mem   | MFU   |
| --------------- | -------------------------- | --------- | ---- | ----- | ----- |
| / 1机8卡（1x8） | / | / | / | / | / |
