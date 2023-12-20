## 模型信息

Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Meta's fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Llama2 outperform open-source chat models on most benchmarks meta's researchers tested, and based on their human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. Meta provide a detailed description of their approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on their work and contribute to the responsible development of LLMs.

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载模型config文件，以及tokenizer。

本测试样例目录下已提供处理好的llama2_7b_hf/目录，相比于huggingface原版，将llama2的max_position_embedding从2048修改为4096。原始的2048是huggingface transformer库的bug

## 数据准备

本测试样例数据准备共分为4个步骤

1. 下载openwebtext原始压缩文件，即：

   https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx 中12GB的openwebtext.tar.xz

2. 全部解压缩

   解压上述12GB的文件后，会出现若干形如urlsf_subsetxxxxxx.xz的压缩文件，将所有压缩文件解压到同一个目录，最终可获得7000000余个txt文件

3. 运行数据预处理文件

   执行preprocess/data_process.py，配置好其中的4个命令行参数。推荐的默认token数量为100M，即1亿个token。此配置在A800 8卡上预计训练1小时

4. 将outputfile（通常为openwebtext_llama2_100M.npy）放置在data_dir下
