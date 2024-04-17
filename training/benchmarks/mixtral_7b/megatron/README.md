## 模型信息
- Introduction

The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mistral-8x7B outperforms Llama 2 70B on most benchmarks we tested.

- Paper
[Mixtral 8x7B](https://arxiv.org/pdf/2401.04088.pdf) 

- 模型代码来源 

This case includes code from the Megatron-LM COMMUNITY LICENSE AGREEMENT License open source project at:https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe


## 数据准备

### 模型配置及tokenizer准备

本测试样例为预训练case，需要下载tokenizer，下载链接为 https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama2/tokenizer.model

在data_dir下创建tokenizer目录，将上述链接中的tokenizer.model文件下载到此目录中


### 数据集准备

本测试样例数据使用FlagScale-llama2预处理好的数据集，下载链接为

https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.bin

https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.idx

在data_dir下创建dataset目录，将上述两个文件放置于data_dir/dataset下。



### 数据集引用

```
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
```