# Qwen1.5-MoE
## 模型信息
- Introduction

Since the surge in interest sparked by Mixtral, research on mixture-of-expert (MoE) models has gained significant momentum. Both researchers and practitioners are keenly interested in understanding how to effectively train such models and assessing their efficiency and effectiveness. Today, Qwen1.5-MoE-A2.7B, a small MoE model with only 2.7 billion activated parameters yet matching the performance of state-of-the-art 7B models like Mistral 7B and Qwen1.5-7B.

Compared to Qwen1.5-7B, which contains 6.5 billion non-embedding parameters, Qwen1.5-MoE-A2.7B contains only 2.0 billion non-embedding parameters, approximately one-third of Qwen1.5-7B’s size. Notably, it achieves a 75% decrease in training expenses and accelerates inference speed by a factor of 1.74, offering substantial improvements in resource utilization without compromising performance.

- Architecture

Researcher build the Qwen1.5-MoE models with a specially designed MoE architecture. Typically, as seen in methods like Mixtral, MoE layers within each transformer block employ eight experts and utilize a top-2 gating strategy for routing purposes. This configuration, while straightforward and efficacious, presents ample scope for enhancement. Consequently, there are some modifications to this architecture:

> Finegrained experts
> 
> Initialization, which is called “upcycling”
>
> Routing mechanism, with shared and routing experts

In terms of model structure, Qwen1.5-MoE introduces a specially designed MoE architecture, optimizes the MoE layer configuration in the existing transformer block, including using 64 finegrained experts, improving the initialization process, and introducing a new Flexible routing mechanism. In particular, the finegrained experts technology generates more experts by dividing the FFN layer, allowing Qwen1.5-MoE to effectively enhance the model's computing power without increasing the number of parameters.

- Performance

Detailed evaluation results are reported in this [blog](https://qwenlm.github.io/blog/qwen-moe/)

Code source [Pai-Megatron-Patch-qwen1.5](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen1_5)

- Details

`Qwen1.5_MoE`共64位专家，采用1.8B基座模型的数据，以4位固定专家+top4路由选择专家的形式激活，共8位专家。`Pai_Qwen1.5_MoE`是`Pai-Megatron-Patch`团队近期（SHA=9fce17b39a958a3455b5cd54b93b3f1f3dc5a5a2）支持的MoE训练方法，采用1.8B基座模型，top2路由选择专家的形式激活，共2位专家，实际*计算量*为`2.3B（计算量=num_hidden_layers * (hidden_size * hidden_size * 4 + hidden_size * intermediate_size * 3 * 路由专家个数) + vocab_size * hidden_size = 24 * (2048 * 2048 * 4 + 2048 * 5632 * 3 * 2)+ 151936 * 2048）`。我们以Pai为蓝本添加的根本原因为目前Pai版本待支持专家细分机制Fine-Grained。Pai版本和qwen版本激活参数量相符，不成为影响mfu计算与芯片评测效果的主要因素。此外，我们结合llm的常见做法，参考Modelscope-Qwen-MoE配置，相比于pai，修改globalbs=512，maxpositionembedding=8192。

Qwen1.5_MoE参数量配置详情见：[Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json)


## 数据准备

### 模型配置及tokenizer准备

本测试样例`预训练`case，需要在[通义千问1.5-1.8B](https://modelscope.cn/models/qwen/Qwen1.5-1.8B/files)下载除模型权重外的其他文件。

在data_dir下创建dataset目录，将上述两个文件放置于data_dir/tokenizer下。

### 数据集准备

本测试样例数据使用FlagScale-llama2预处理好的数据集，下载链接为

https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.bin

https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.idx

在data_dir下创建llama_00_text_document目录，将上述两个文件放置于data_dir/llama_00_text_document下。

### 目录树

#### Dataset 数据集
```
└── datasets
    └── qwen1.5_14B_MoE
        ├── llama_00_text_document
        │   ├── llama_00_text_document
        │   ├── llama_00_text_document.bin
        │   └── llama_00_text_document.idx
        └── tokenizer
            ├── LICENSE
            ├── README.md
            ├── config.json
            ├── configuration.json
            ├── generation_config.json
            ├── merges.txt
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── vocab.json
```

### 数据集引用

```
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
```