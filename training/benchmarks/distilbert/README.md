## Model Introduction
### DistilBERT base model (uncased)

This model is a distilled version of the [BERT base model](https://huggingface.co/bert-base-uncased). It was
introduced in [this paper](https://arxiv.org/abs/1910.01108). The code for the distillation process can be found
[here](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation). This model is uncased: it does
not make a difference between english and English.

## Model and Training Scripts source code
Pytorch case:
This repository includes software from https://github.com/huggingface/transformers/tree/v4.33.0
licensed under the Apache License 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

## Dataset and Model Checkpoints

> Dataset website：https://huggingface.co/datasets/sst2
https://huggingface.co/distilbert-base-uncased
> Model checkpoint website: https://huggingface.co/distilbert-base-uncased

We have already preprocessed the dataset and the model checkpoint files(The preprocessing script is `training/benchmarks/distilbert/pytorch/data_preprocessing/create_train_eval_data.py`).
The preprocessed can be downloaded directly from https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/datasets/distilbert_train.tar.
No additional preprocessing steps need to be conducted.

After decompressing, the dataset and model checkpoint files are organized as the following:

```
distilbert
├── dataset                     # dataset files
│   ├── eval_dataset.npz
│   └── train_dataset.npz
└── model                       # model checkpoint and config files
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

## Benchmark Task and Target Accuracy
This experiment is to finetune a text classification task on SST-2 dataset with DistilBERT-base-uncased pretrained checkpoints.
After finetuning 10 epoches, the DistilBERT-base-uncased model is able to achieve accuracy score of 90+, which matches the evaluation result on the [report](https://huggingface.co/distilbert-base-uncased).

## AI Frameworks && Accelerators supports

|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | [✅](../../nvidia/distilbert-pytorch/README.md)       | N/A    | N/A       |
