## 模型信息
### 模型介绍

BERT stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

Please refer to this paper for a detailed description of BERT:
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### 模型Checkpoint下载
[模型Checkpoint下载](../../benchmarks/bert/pytorch/readme.md#模型信息与数据集模型checkpoint下载)
### 测试数据集下载
[测试数据集下载](../../benchmarks/bert/pytorch/readme.md#模型信息与数据集模型checkpoint下载)

### 天数智芯 BI-V100 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本:  4.15.0-156-generic x86_64    
   - 加速卡驱动版本：3.0.0
   - Docker 版本：20.10.8
   - 训练框架版本：torch-1.10.2+corex.3.0.0
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件            | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s) |
| -------- | ------------------ | ---------- | ------- | -------  | ------- | --------------- |
| 单机1卡  | config_BI-V100x1x1 | 17854.76    | 0.72    | 0.7325   | 25000   |17.00            |
| 单机8卡  | config_BI-V100x1x8 | 20312.57    | 0.72    | 0.9619   | 25000   |118.45           |
| 两机8卡  | config_BI-V100x2x8 | pending     | 0.72    | pending  | pending |pending          |

### 许可证

本项目基于Apache 2.0 license。

本项目部分代码基于MLCommons https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA 实现。