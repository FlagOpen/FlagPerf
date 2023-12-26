### 昆仑芯XPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
  - 机器型号: 昆仑芯AI加速器组R480-X8
  - 加速卡型号: 昆仑芯AI加速卡R300
  - CPU型号: Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：4.23
  - Docker镜像和版本：xpytorch/kunlunxin-deepspeed:v1.0
  - 训练框架版本：XPyTorch 2.0.1 + deepspeed 0.10.1
  - 依赖软件版本：transformers 4.32.1

- ##### 并行策略

   - 并行技术：sharded data parallel
   - 实施者：deepspeed ZeRO-DP
   - 实施细节：ZeRO-DP O3, DP_SIZE=8

- ##### 优化策略

  - gradient_checkpointing
  - FC算子的分块策略调优



### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A100x1x8.py中所写，在本case中默认为1
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A100x1x8.py中所写，在本case中默认为4096
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为ds_config.json中所写，在本case中默认为1
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size，简写为GBS。在本case中，只存在数据并行，因此data_parallel_size=world_size。

* 通用指标

| 指标名称       | 指标值                  | 特殊说明                                    |
| -------------- | ----------------------- | ------------------------------------------- |
| 任务类别       | 自然语言理解            |                                             |
| 模型           | deepspeed-llama2-7b     |                                             |
| 数据集         | openwebtext             | 如无特殊说明，训练前1亿个token              |
| 数据精度       | fp16                    |                                             |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参          |
| 硬件设备简称   | R300                    |                                             |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB                    |
| 吞吐量         | token/p/s,见“性能指标”  | 平均单卡每秒处理的token数                   |
| 损失值         | loss,见“性能指标”       | 训练损失值                                  |
| 计算使用率     | MFU,见“性能指标”        | 参见PaLM论文定义                            |

* 性能指标

| 配置                | fix_hp              | tokens/p/s | loss  | mem     |   MFU  |
| ------------------- | ------------------- | --------   | ----- | ------- | ------ |
| R300单机8卡（1x8）  | MPE=512 LBS=12      |            |  5.27 | 29/32   |        |


