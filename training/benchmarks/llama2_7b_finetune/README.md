## 模型信息

Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Meta's fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Llama2 outperform open-source chat models on most benchmarks meta's researchers tested, and based on their human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. Meta provide a detailed description of their approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on their work and contribute to the responsible development of LLMs.

代码来源: https://github.com/facebookresearch/llama-recipes/tree/main

## 模型配置及tokenizer准备

本测试样例为微调case，需要下载模型config文件，以及tokenizer还有模型权重，

下载链接为：https://huggingface.co/meta-llama/Llama-2-7b-hf

下载后路径填写至【model_name】 字段，具体位置FlagPerf/training/nvidia/llama2_7b_finetune-pytorch/config/config_A100x1x1.py

需要将llama2的max_position_embedding从2048修改为实际使用值。本项目中Nvidia版本代码为512。

## 数据准备

### SFT训练和验证数据集
本模型训练验证数据集为samsum_dataset，

下载链接：https://huggingface.co/datasets/samsum/resolve/main/data/corpus.7z

下载解压完成后，命名文件夹为 samsum_dataset，并放置在FlagPerf/training/run_benchmarks/config/test_conf.py 中对应配置路径的子目录下，例如如下xxxx目录指向的路径中：
    "llama2_7b_finetune:pytorch_2.0.1:A100:1:1:1": "XXXXX",

### 评估数据集

* 本模型验证数据集为MMLU

* 下载地址：`https://huggingface.co/datasets/Stevross/mmlu/tree/main`
  1. 下载其中的data.tar
  2. 将.tar文件还原为目录
  3. 下载解压完成后，命名文件夹为 mmlu_dataset，并放置在FlagPerf/training/run_benchmarks/config/test_conf.py 中对应配置路径的子目录下，例如xxxx目录中：
    "llama2_7b_finetune:pytorch_2.0.1:A100:1:1:1": "XXXXX",
