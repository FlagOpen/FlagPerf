## 模型信息

Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Meta's fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Llama2 outperform open-source chat models on most benchmarks meta's researchers tested, and based on their human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. Meta provide a detailed description of their approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on their work and contribute to the responsible development of LLMs.

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载tokenizer。

本测试样例目录下已提供tokenizer.json (来自https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main) 在preprocess目录

## 数据准备

本测试样例数据准备共分为4个步骤

1. 下载RedPajama-Data-1T-Sample原始数据，即：

   https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/tree/main 中所有的.jsonl文件，存在某一文件夹下，假定为data_dir

2. 预处理

   cd preprocess && bash preprocess.sh data_dir your_preprocessed_dir/RedPajama-Data-1T-Sample

3. 设置training/run_benchmarks/config/test_config.py对应模型的数据目录