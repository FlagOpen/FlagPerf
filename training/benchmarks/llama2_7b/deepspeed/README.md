## 模型信息

Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Meta's fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Llama2 outperform open-source chat models on most benchmarks meta's researchers tested, and based on their human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. Meta provide a detailed description of their approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on their work and contribute to the responsible development of LLMs.

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载模型config文件，以及tokenizer。

本测试样例目录下已提供处理好的llama2_7b_hf/目录，相比于huggingface原版，将llama2的max_position_embedding从2048修改为4096。原始的2048是huggingface transformer库的bug

## 数据准备

当前目录的data/目录下，存放着数据


### 昆仑芯XPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
  - 机器型号: 昆仑芯AI加速器组R480-X8
  - 加速卡型号: 昆仑芯AI加速卡R300
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：4.0.25
  - Docker镜像和版本：pytorch2.0.1-cu17-ubuntu20.04:v0.01
  - 训练框架版本：xmlir
  - 训练编译器版本：xacc
  - 依赖软件版本：pytorch-2.0.1+cu17

### 运行命令
 bash run_llama.sh

### 运行情况

* 通用指标

| 指标名称       | 指标值                  | 特殊说明                                    |
| -------------- | ----------------------- | ------------------------------------------- |
| 任务类别       | 自然语言理解            |                                             |
| 模型           | deepspeed-llama2-7b      |                                             |
| 数据集         |         openwebtext      |                                             |
| 数据精度       | precision,见“性能指标”  | 可选fp32/amp/fp16                           |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参          |
| 硬件设备简称   | R300             |                                             |
| 硬件存储使用   | memory,见“性能指标”        | 通常称为“显存”,单位为GiB                    |
| 吞吐量     | token/p/s,见“性能指标”   | 平均单卡每秒处理的token数                     |
| 损失值       | loss,见“性能指标”    | 训练损失值 |
| 计算使用率   | MFU,见“性能指标”    | 参见PaLM论文定义           |

* 性能指标

| 配置                | precision | fix_hp              | tokens/p/s | loss  | memory | MFU |
| ------------------- | --------- | ------------------- | --------   | ----- | ------- | ------ | ------- | --------- |
| R300单机8卡（1x8）  |  fp32     | bs=1,seqlength=512  |           |  5.4  |         |        |         |
| R300单机8卡（1x8）  |  fp32     | bs=2,seqlength=512  |            |  5.4  |         |
| R300单机8卡（1x8）  |  fp16     | bs=1,seqlength=512  |            |  5.99 |         |        |


