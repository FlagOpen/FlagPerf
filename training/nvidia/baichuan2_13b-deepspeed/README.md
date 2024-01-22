### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### H800硬件环境
    - 机器型号: NVIDIA H800(80G)
    - 加速卡型号: NVIDIA_H800-80GB
    - CPU型号: Intel(R) Xeon(R) Platinum 8462Y+
    - 多机网络类型、带宽: InfiniBand, 200Gb/s

- ##### H800软件环境
   - OS版本：Ubuntu 22.04 LTS
   - OS kernel版本: 5.15.0-25-generic     
   - 加速卡驱动版本：535.129.03
   - Docker 版本：24.0.7
   - 训练框架版本：deepspeed 0.11.1
   - 依赖软件版本：sentencepiece

- ##### 并行策略

   - 并行技术：sharded data parallel
   - 实施者：deepspeed ZeRO-DP
   - 实施细节：ZeRO-DP O1, DP_SIZE=16

- ##### 优化策略

   - gradient checkpointing

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_H800x2x8.py中所写，在本case中默认为1
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_H800x2x8.py中所写，在本case中默认为2048
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
| 硬件设备简称  | nvidia H800           |                               |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB               |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                    |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数               |

* 性能指标

| 配置              | fix_hp                     | token/p/s | loss | mem   | MFU   |
| --------------- | -------------------------- | --------- | ---- | ----- | ----- |
| H800 2机16卡（2x8） | /                          | 2068      | 2.06 | 77/80 | 16.3% |
| H800 2机16卡（2x8） | GAS=64(globalbs=2M tokens) | 2710      | 2.07 | 77/80 | 21.4% |

* 补充说明

经排查，baichuan2原生模型未添加flashattention2的支持。flashattention2可以极大的降低显存开销，并降低互联需求。llama2，Aquila2等模型均支持此方案。此次实验只能在未支持flashattention2的情况下，开启gradient checkpointing（使用34%的额外计算量来节约显存）。作为对比，llama2-7B在A100机器MFU可以达到约50%，不开启flashattention2可以达到约28%，不开启flashattention2且开启gradient checkpointing能达到21%，同结构的llama2-70B在H800上开启flashattention2且关闭gradient checkpointing也可以达到约40%。

因此参数量变化不是导致baichuan2-13B MFU低的问题，关键原因是baicuan2-13B不支持目前主流算法通用的flashattention2算子，导致计算速度慢且显存开销大，不开启gradient checkpointing会OOM，因为gradient checkpointing是用额外计算换显存，拿时间换空间，所以性能相比仅不开启flashattention2进一步降低。综上，本次实验21.4%的MFU正常。

附表：各类实验MFU对比（均已在并行方案等其他可能影响性能的配置上达到较优）

| 机器+模型              | 正常配置 | 不开启flashattention | 不开启flashattention且开启gradient checkpointing |
| ------------------ | ---- | ----------------- | ---------------------------------------- |
| A100+llama2-7B     | 50%  | 28%               | 21%                                      |
| H800+llama2-70B    | 40%  | /                 | /                                        |
| H800+baichuan2-13B | /    | /                 | 21.4%                                    |
