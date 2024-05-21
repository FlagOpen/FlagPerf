## Model Introduction
### longformer-base-4096 model

longformer-base-4096 is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents.
It supports sequences of length up to 4,096. [longformer-base-4096 model](https://huggingface.co/allenai/longformer-base-4096). It was
introduced in [this paper](https://arxiv.org/abs/2004.05150). The code for longformer process can be found
[here](https://github.com/huggingface/transformers/tree/main/examples/research_projects/longform-qa).

## Model and Training Scripts source code
Pytorch case:
This repository includes software from https://github.com/huggingface/transformers/tree/v4.33.0
licensed under the Apache License 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

## Dataset and Model Checkpoints

> Dataset website：https://huggingface.co/datasets/enwik8 and https://huggingface.co/allenai/longformer-base-4096
> Model checkpoint website: https://huggingface.co/allenai/longformer-base-4096

We have already preprocessed the dataset and the model checkpoint files(The preprocessing script is `training/benchmarks/longformer/pytorch/data_preprocessing/create_train_eval_data.py`).
The preprocessed dataset can be downloaded directly from https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/datasets/longformer_train.tar.
No additional preprocessing steps need to be conducted.
After decompressing, the dataset and model checkpoint files are organized as the following:

```
longformer_train
|-- dataset
|   |-- eval_dataset.npz
|   `-- train_dataset.npz
`-- model
    |-- config.json
    |-- merges.txt
    |-- pytorch_model.bin
    |-- tokenizer.json
    `-- vocab.json
```

## Benchmark Task and Target Accuracy

This experiment is to finetune a text classification task on wikitext dataset with longformer-base-4096 pretrained checkpoints.
After finetuning 10 epoches, the longformer-base-4096 model is able to achieve accuracy score of 90+, which matches the evaluation result on the [report](https://huggingface.co/allenai/longformer-base-4096).

## AI Frameworks && Accelerators supports

|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | [✅](../../nvidia/longformer-pytorch/README.md)       | N/A    | N/A       |
