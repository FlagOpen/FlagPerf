### 模型信息
#### 模型介绍
We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community1.

Please refer to this paper for a detailed description of LLaMA1: 
[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) 

#### 模型代码来源
Paddle case代码来源:
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/llama licensed under the Apache License, Version 2.0.

#### 数据集
##### 测试数据集下载地址
测试数据集中提供了处理好的openwebtext 100k条 doc的训练样本：
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
```

##### 预处理
> 无需预处理 

#### 模型实现
* 运行自动加载

#### 模型checkpoint
* 运行自动下载
* Paddle的 LLaMA 模型的权重的使用则需要遵循[License](../../paddlenlp/transformers/llama/LICENSE)。

### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU |N/A  |✅  |N/A|
|    |   |    |   |