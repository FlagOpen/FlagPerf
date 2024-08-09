
### Nvidia GPU配置与运行信息参考
#### A100环境配置
- ##### 硬件环境

    - 机器、加速卡型号: 曦云®C500 64G
    - 多机网络类型、带宽: InfiniBand，2x200 Gb/s

- ##### 软件环境

   - OS版本：Ubuntu 20.04.6
   - OS kernel版本:  5.4.0-26-generic
   - 加速卡驱动版本：2.3.0
   - Docker 版本：24.0.7
   - 训练框架版本：pytorch-2.0.0+mc2.23.0.13-cp38-cp38-linux_x86_64.whl, megatron-core 0.6.0
   - 依赖软件版本：sentencepiece==0.2.0, transformers==4.40.1

- ##### 并行策略

   - 并行技术：张量、流水、数据混合并行，具体并行方案见“运行情况”章节
   - 实施者：megatron-core
   - 实施细节：PP2DP2TP2

- ##### 优化策略

   - flash attention 2
   - recompute-activations
   - transformer-engine local

### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_A100x1x8.py中所写，在本case中默认为1。
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
| 硬件设备简称  | Metax C500           |                                          |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB                          |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                               |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数                          |

* 性能指标

精度对齐需第21步及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。NVloss曲线请联系智源研究院获取

| 配置                 | precision | parallel    | fix_hp | token/p/s | 是否精度对齐     | mem   | MFU         |
| ------------------ | --------- | ----------- | ------ | --------- | ---------- | ----- | ----------- |
| C500单机8卡（1x8）      | bf16      | PP2DP2TP2   | /      | /    | True | 51.6/64.0 | 40.4%       |
