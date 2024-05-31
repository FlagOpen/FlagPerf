### 模型信息

- BERT stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
  BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

  Please refer to this paper for a detailed description of BERT:
  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


This repository includes software from https://huggingface.co/docs/transformers/training
licensed under the Apache License, Version 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

### 数据集

#### 数据集下载地址

 ● 下载地址：`https://model.baai.ac.cn/model-detail/100097`，下载其中的：

```
文件列表：
openwebtext_bert_10M.npy
openwebtext_bert_100M.npy
```

* 解压后将两个文件放置在<data_dir>目录下

#### 预训练权重 

* 模型实现
  * pytorch：transformers.BertForMaskedLM
* 权重下载
  * pytorch：BertForMaskedLM.from_pretrained("bert-large/base-uncased")
* 权重选择
  * 使用save_pretrained将加载的bert-large或bert-base权重保存到<data_dir>/<weight_dir>路径下

