
### Nvidia GPU配置与运行信息参考
#### A100环境配置
- ##### 硬件环境

    - 机器型号: NVIDIA DGX A800(80G) 
    - 加速卡型号: NVIDIA_A800-SXM4-80GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境

   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-126-generic     
   - 加速卡驱动版本：470.141.10
   - Docker 版本：20.10.18
   - 训练框架版本：megatron-core tag:core_v0.6.0
   - 依赖软件版本：sentencepiece==0.2.0, transformers==4.40.1

- ##### 并行策略

   - 并行技术：张量、流水、数据混合并行，具体并行方案见“运行情况”章节
   - 实施者：megatron-core
   - 实施细节：PP2DP4TP1

- ##### 优化策略

   - flash attention 2
   - recompute-activations
   - transformer-engine impl

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A100x1x8.py中所写，在本case中默认为1。**厂商适配时可任意更改**
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A100x1x8.py中所写，在本case中默认为8192，原则上不可更改
  3. global_batchsize恒等于local_batchsize\*gradient_accumulate_steps\*data_parallel_size。在本case中，data_parallel_size=world_size/TPsize/PPsize。在本case中默认为512，使得globalbatchsize=4M tokens。

* 通用指标

| 指标名称    | 指标值                   | 特殊说明                                     |
| ------- | --------------------- | ---------------------------------------- |
| 任务类别    | 自然语言理解                |                                          |
| 模型      | llama3_8b             |                                          |
| 数据集     | wudao                 | wudao数据集来源于智源研究院<br>bin/idx数据集文件来源于阿里云灵骏团队<br>使用llama3 tokenizer预处理 |
| 数据精度    | precision,见“性能指标”     | 可选fp32/amp/fp16/bf16                     |
| 超参修改    | parallel,见“性能指标”      | 格式为PPxDPyTPz，例如PP2DP4TP1                 |
| 超参修改    | fix_hp,见“性能指标”        | 跑满硬件设备评测吞吐量所需特殊超参                        |
| 硬件设备简称  | nvidia A100           |                                          |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB                          |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                               |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数                          |

* 性能指标

精度对齐需第21步及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。NVloss曲线请联系智源研究院获取

| 配置                 | precision | parallel    | fix_hp | token/p/s | 是否精度对齐     | mem   | MFU         |
| ------------------ | --------- | ----------- | ------ | --------- | ---------- | ----- | ----------- |
| A100单机8卡（1x8）      | bf16      | PP2DP4TP1   | /      | 3698.7    | True(作为基线) | 76/80 | 56.9%       |
| H100单机8卡（1x8）      | bf16      | PP1DP4TP2   | /      | 8103      | True       | 71/80 | 39.3%       |
| H100单机8卡（1x8）      | bf16+fp8  | PP1DP4TP2   | /      | 10486     | True       | 74/80 | 25.4%-50.6% |
| H100二机16卡（2x8）     | bf16      | PP1DP8TP2   | /      | 7914      | True       | 62/80 | 38.4%       |
| H100二机16卡（2x8）     | bf16+fp8  | PP1DP8TP2   | /      | 10280     | True       | 64/80 | 24.9%-49.9% |
| H100-55机440卡（55x8） | bf16      | PP1DP220TP2 | gbs=1760*      | 7047      | /,仅供性能参考       | 55/80 | 34.2%       |
| \*单机8卡（1x8）      | bf16      | PP2DP4TP1   | /      | 2260      | True       | \* | \*       |

*A100单机8卡 消融实验*
| 配置            | precision | parallel  | fix_hp | token/p/s | 是否精度对齐 | mem   | MFU   |
| ------------- | --------- | --------- | ------ | --------- | ------ | ----- | ----- |
| A100单机8卡（1x8） | bf16      | PP4DP2TP1 | /      | 3306.5    | /      | 59/80 | 50.0% |
| A100单机8卡（1x8） | bf16      | PP1DP8TP1 | /      | /         | /      | OOM   | /     |
| A100单机8卡（1x8） | bf16      | PP2DP2TP2 | /      | 3409.2    | /      | 51/80 | 52.4% |
| A100单机8卡（1x8） | bf16      | PP2DP1TP4 | /      | 3006.1    | /      | 35/80 | 46.2% |
| A100单机8卡（1x8） | bf16      | PP8DP1TP1 | /      | 2690.3    | /      | 55/80 | 41.4% |
| A100单机8卡（1x8） | bf16      | PP1DP1TP8 | /      | 2451.2    | /      | 30/80 | 37.7% |
| A100单机8卡（1x8） | bf16      | PP4DP1TP2 | /      | 3042.7    | /      | 45/80 | 46.8% |

OOM: Out Of Memory

注：H100标定bf16算力为989TFLOPS，fp8算力为1979TFLOPS。因此bf16+fp8混合精度的MFU位于两者之间，为一个范围。如全部为bf16则MFU取上界，如全部转为FP8则MFU取下界。如需使用fp8数制进行训练，可以将training\_adapter\_H100\_fp8.sh中的内容拷贝到training\_adapter.sh中。注意：只有英伟达H及更高系列GPU等少数AI芯片具备FP8计算能力。
