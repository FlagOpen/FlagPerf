## 模型信息
### 模型介绍

Generative Pre-trained Transformer 3 (GPT-3) is a large language model released by OpenAI in 2020. Like its predecessor GPT-2, it is a decoder-only transformer model of deep neural network, which uses attention in place of previous recurrence- and convolution-based architectures. Attention mechanisms allow the model to selectively focus on segments of input text it predicts to be the most relevant. It uses a 2048-tokens-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model demonstrated strong zero-shot and few-shot learning on many tasks.

Please refer to wikipedia for a detailed description of GPT-3:
[GPT-3 (wikipedia)](https://en.wikipedia.org/wiki/GPT-3)


###  模型代码来源
Paddle Case
This repository includes software from https://github.com/PaddlePaddle/PaddleNLP licensed under the Apache License, Version 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

### 模型Checkpoint下载

* 模型实现
  * 运行自动下载
* 权重下载
  * 运行自动下载

### 测试数据集下载

Paddle Case

```shell
mkdir data-gpt3
cd data-gpt3 # 后续在training/run_benchmarks/config/test_conf.py中修改数据位置使用
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

### 框架与芯片支持情况
* GPT-3 (6.7B)
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | N/A |[✅](../../nvidia/gpt3-6.7b-paddle/README.md)  |N/A|


* GPT-3 (13B)
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | N/A |[✅](../../nvidia/gpt3-13b-paddle/README.md)  |N/A|
