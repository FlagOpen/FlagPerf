## 模型信息
- Introduction

Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Meta's fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Llama2 outperform open-source chat models on most benchmarks meta's researchers tested, and based on their human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. Meta provide a detailed description of their approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on their work and contribute to the responsible development of LLMs.

- Paper
[LLAMA2](https://arxiv.org/pdf/2307.09288.pdf) 

- 模型代码来源 

This case includes code from the LLAMA 2 COMMUNITY LICENSE AGREEMENT License open source project at:https://github.com/facebookresearch/llama-recipes/tree/main


## 数据准备

### 模型配置及tokenizer准备

本测试样例为预训练case，需要下载tokenizer，下载链接为 https://huggingface.co/meta-llama/Llama-2-70b-hf，注意需要自行向meta申请下载授权。

在data_dir下创建tokenizer目录，将上述链接中的tokenizer.model文件下载到此目录中


### 数据集准备

本测试样例数据使用FlagScale-llama2预处理好的数据集，下载链接为

https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin

https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx

将上述两个文件放置于data_dir下。

This case includes datasets from the MIT License open source project at https://github.com/EleutherAI/the-pile

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

### 数据集引用

```
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
```

### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- |
| Nvidia GPU | ✅       |
| 昆仑芯 XPU | N/A     |
| 天数智芯   | N/A     |