# 模型信息
- Introduction

The `Mixtral-8x7B` Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The first batch of publicly released models includes two sizes: 8x7B and 8x22B parameters. 

We used Megatron Core-v0.6.0 and implemented the `Mixtral-8x7B` pre-training computation task based on the Megatron framework, following the algorithm architecture and configurations of `Mixtral-8x7B`.

[Technical Blog](https://mistral.ai/news/mixtral-of-experts/)(there's no paper until May. 13th, 2024) 

- 模型代码来源

Mistral AI team仅开源了`Mixtral-8x7B`的模型权重文件、tokenizer等，不包括预训练代码、预训练实现、预训练数据集等。出于上述事实，本评测样例基于开源Megatron框架，使用开源wudao数据集，在[mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) · Hugging Face设计的Mixtral-8x7B算法结构上进行预训练，来进行AI硬件评测。测试样例代码为FlagPerf编写。需要下载或准备的文件见数据准备小节，依赖的外部软件或信息见依赖小节。

# 数据准备

### 模型配置及tokenizer准备

本测试样例为预训练case，需要下载tokenizer（如不确定tokenizer包含哪些文件，可下载Mixtral-8x7B-v0.1的所有文件，尽管我们不需要使用模型权重文件），不需要下载模型代码或模型权重。tokenizer需在data_dir下创建tokenizer目录，按照huggingface要求的格式进行处理或存放。了解data\_dir需要阅读FlagPerf有关训练的文档，或直接修改FlagPerf/training/run_benchmarks/config/test_conf.py中CASES变量中的value。

### 数据集准备

本测试样例数据使用智源研究院wudao数据集，使用Mixtral提供的tokenizer并进行预处理。测试数据集的内容不是影响AI硬件评测结果的核心因素，可以参考或使用阿里灵骏团队开放的预处理好的数据集[Pai-Megatron-Patch/examples/mistral/README.md at main · alibaba/Pai-Megatron-Patch (github.com)](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/mistral/README.md)

在上述README文件中找到并执行

```
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-valid.json

```

将上述4个文件（不清楚文件原理可以参考Megatron-LM仓库相关介绍）放置于data_dir下。

### wudao数据集（数据集内容）

[BAAI-WuDao/WuDaoMM: WuDaoMM this is a data project (github.com)](https://github.com/BAAI-WuDao/WuDaoMM?tab=readme-ov-file)

### 灵骏团队（megatron格式数据文件，供参考）

[alibaba/Pai-Megatron-Patch: The official repo of Pai-Megatron-Patch for LLM & VLM large scale training developed by Alibaba Cloud. (github.com)](https://github.com/alibaba/Pai-Megatron-Patch/tree/main)

### megatron-core（计算任务实现）

[NVIDIA/Megatron-LM: Ongoing research training transformer models at scale (github.com)](https://github.com/NVIDIA/Megatron-LM/tree/main?tab=License-1-ov-file#readme)
