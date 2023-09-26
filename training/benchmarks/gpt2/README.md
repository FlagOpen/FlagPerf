### 模型信息
- 模型介绍

GPT-2 Medium is the 345M parameter version of Megatron-GPT2, a transformer-based language model created and released by OpenAI. The model is a pretrained model on English language using a causal language modeling (CLM) objective.

>[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 

- 模型代码来源

This case includes code from open source project at https://github.com/NVIDIA/Megatron-LM/tree/v3.0/megatron

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.


### 数据集
- 数据集下载地址
> Dataset website：https://huggingface.co/datasets/lambada

> The training data should be downloaded from huggingface. First, download training data in a loose json format, with one json containing a text sample per line. For example in python interpreter:

```
from datasets import load_dataset

train_data = load_dataset('lambada', split='train')
train_data.to_json("lambada.train.json", lines=True)
```

- 预处理
> The training data requires preprocessing. 
The loose json is then processed into a binary format for training. To convert the json into mmap format use preprocess_data.py. An example script to prepare data for GPT2 training is:

``` bash
python tools/preprocess_data.py \
        --input lambada.train.json \
        --output-prefix lambada \
        --vocab gpt2-vocab.json \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file gpt2-merges.txt \
        --append-eod \
        --workers 32 \
        --chunk-size 25 \
```


### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
