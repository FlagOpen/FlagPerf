# 模型信息
- Introduction

The `Mixtral-8x7B` Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The first batch of publicly released models includes two sizes: 8x7B and 8x22B parameters. 

We used Megatron Core-v0.6.0 and implemented the `Mixtral-8x7B` pre-training computation task based on the Megatron framework, following the algorithm architecture and configurations of `Mixtral-8x7B`.

[Technical Blog](https://mistral.ai/news/mixtral-of-experts/)(there's no paper until May. 13th, 2024) 

- 模型代码来源

Mistral AI team仅开源了`Mixtral-8x7B`的模型权重文件、tokenizer等，不包括预训练代码、预训练实现、预训练数据集等。出于上述事实，本评测样例基于开源Megatron框架，使用开源wudao数据集，在[mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) · Hugging Face设计的Mixtral-8x7B算法结构上进行预训练，来进行AI硬件评测。测试样例代码为FlagPerf编写。需要下载或准备的文件见数据准备小节，依赖的外部软件或信息见依赖小节。


# 数据准备

### 模型配置及tokenizer准备



### 数据集准备


在上述README文件中找到并执行


将上述两个文件（.bin与.idx，不清楚文件原理可以参考Megatron-LM仓库相关介绍）放置于data_dir下。

# 依赖

