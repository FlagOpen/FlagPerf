### 沐曦集成电路 C500 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: 曦云®C500 64G 
    - 多机网络类型、带宽: InfiniBand，2x200 Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04.6
   - OS kernel版本:  5.4.0-26-generic
   - 加速卡驱动版本：2.2.0
   - Docker 版本：24.0.7
   - 训练框架版本：pytorch-2.0.0+mc2.19.2.23-cp38-cp38-linux_x86_64.whl, deepspeed 0.10.0

- ##### 并行策略
   - 并行技术：sharded data parallel
   - 实施者：deepspeed ZeRO-DP
   - 实施细节：ZeRO-DP O2, ZeRO-Offload, DP_SIZE=16

- ##### 优化策略
   - gradient checkpointing

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_C500x2x8.py中所写，在本case中默认为4
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_C500x2x8.py中所写，在本case中默认为2048
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为ds_config.json中所写，在本case中默认为16，精度对齐实验默认为64
  4. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size，简写为GBS。在本case中，只存在数据并行，因此data_parallel_size=world_size。

* 通用指标

| 指标名称    | 指标值                   | 特殊说明                          |
| ------- | --------------------- | ----------------------------- |
| 任务类别    | 自然语言理解                |                               |
| 模型      | baichuan2_13b         |                               |
| 数据集     | openwebtext           | 如无特殊说明，训练前1亿个token            |
| 数据精度    | bf16                  |                               |
| 超参修改    | fix_hp,见“性能指标”        | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称  | Metax C500           |                               |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB               |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                    |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数               |

* 性能指标

| 配置              | fix_hp                     | token/p/s | loss | mem   | MFU   |
| --------------- | -------------------------- | --------- | ---- | ----- | ----- |
| C500 2机16卡（2x8） | local_batchsize=4， GAS=16(globalbs=2M tokens) | /     | 2.07 | / | / |

* 补充说明

Metax C500 在local_batchsize=1， GAS=64的配置下进行了精度测试，在此配置下 Metax C500 的 Loss 与 Nvidia H800 完全对齐，上述测试case的global_batchsize与精度测试一致。