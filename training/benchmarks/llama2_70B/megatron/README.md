## 模型信息

Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Meta's fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Llama2 outperform open-source chat models on most benchmarks meta's researchers tested, and based on their human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. Meta provide a detailed description of their approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on their work and contribute to the responsible development of LLMs.

代码来源: https://github.com/facebookresearch/llama-recipes/tree/main

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载tokenizer，下载链接为 https://github.com/FlagOpen/FlagScale/tree/main/examples/llama2/tokenizer

在data_dir下创建tokenizer目录，将上述链接中的tokenizer.model文件下载到此目录中

## 数据准备

本测试样例数据使用FlagScale-llama2预处理好的数据集，下载链接为

<!-- https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin

https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx -->

将上述两个文件放置于data_dir下。