
## Model Introduction
### What is T5-Small(Text-To-Text Transfer Transformer)?
The developers of the Text-To-Text Transfer Transformer (T5) [write](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html):

> With T5, we propose reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input. Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task.

T5-Small is the checkpoint with 60 million parameters.

- **Developed by:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. See [associated paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf) and [GitHub repo](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints)
- **Model type:** Language model
- **Language(s) (NLP):** English, French, Romanian, German
- **License:** Apache 2.0
- **Related Models:** [All T5 Checkpoints](https://huggingface.co/models?search=t5)
- Resources for more information:
  - [Research paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
  - [Google's T5 Blog Post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
  - [GitHub Repo](https://github.com/google-research/text-to-text-transfer-transformer)
  - [Hugging Face T5 Docs](https://huggingface.co/docs/transformers/model_doc/t5)

## Model and Training Scripts source code
Pytorch case:
This repository includes software from https://github.com/huggingface/transformers/blob/v4.31.0/examples/pytorch/summarization/run_summarization_no_trainer.py
licensed under the Apache License 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

## Dataset and Model Checkpoints

> Dataset website：https://huggingface.co/datasets/cnn_dailymail and https://github.com/abisee/cnn-dailymail

> Model checkpoint website: https://huggingface.co/t5-small/tree/main

We have already preprocessed the dataset and the model checkpoint files(The preprocessing script is `training/benchmarks/t5_small/pytorch/data_preprocessing/create_train_eval_data.py`).
The preprocessed can be downloaded directly from https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/datasets/t5_small_train.tar.
No additional preprocessing steps need to be conducted.

After decompressing, the dataset and model checkpoint files are organized as the following:

```
t5_small_train
├── dataset                     # dataset files
│   ├── eval_dataset.npz
│   └── train_dataset.npz
├── metrics                     # metrics for evaluation
│   └── rouge
│       └── rouge.py
├── model                       # model checkpoint and config files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── nltk_data                   # nltk data for evaluation
    └── tokenizers
        └── punkt
```

## Benchmark Task and Target Accuracy
This experiment is to finetune a summarization task on CNN/Daily Mail dataset with t5-small pretrained checkpoints.
After finetuning 3 epoches, the t5-small model is able to achieve a ROUGE-1 score of 41+, which matches the evaluation result on the [paper](https://arxiv.org/abs/1910.10683).

## AI Frameworks && Accelerators supports

|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | [✅](../../nvidia/t5_small-pytorch/README.md)       | N/A    | N/A       |
