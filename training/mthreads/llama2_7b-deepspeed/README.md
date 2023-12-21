### Moore Threads S4000 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器型号: MCCX D800 
    - 加速卡型号: S4000
    - CPU型号: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
    - 多机网络类型、带宽: InfiniBand，2x200Gb/s
    
- ##### 软件环境
   - OS版本：Ubuntu 20.04 LTS
   - OS kernel版本: 5.4.0-42-generic
   - 加速卡驱动版本：2.2.0
   - Docker镜像和版本: PyTorch2.0_musa1.4_ec6a747fd342 
   - 训练框架版本：pytorch-2.0.0+torch_musa-git8ea3501
   - 依赖软件版本:
     - musa toolkits: 1.4.0+git4e25703
     - mublas: 1.1.0+gite484aa2

- ##### 优化策略

   - scaled dot product attention
   - checkpointing

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_S4000x1x8.py中所写，在本case中默认为3
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_S4000x1x8.py中所写，在本case中默认为4096
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为ds_config.json中所写，在本case中默认为1
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size，简写为GBS。在本case中，只存在数据并行，因此data_parallel_size=world_size。

* 通用指标

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 自然语言理解               |                                    |
| 模型         | llama2_7b                  |                                    |
| 数据集       | openwebtext                | 如无特殊说明，训练前1亿个token |
| 数据精度     | amp                        |                                    |
| 超参修改     | fix_hp,见“性能指标”        | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称 | S4000                      |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |

* 性能指标

| 配置                |  fix_hp           | token/p/s | loss | mem       | MFU       |
| ------------------- | ---------------- | ------ | ------- | --------- | --------- |
| S4000单机8卡（1x8）  |       / |  |44.2/48.0|3.20|  |
