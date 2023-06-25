### 模型信息
- Introduction

  Graph Convolutional Networks (GCNs) have recently been shown to be quite successful in modeling graph-structured data. However, the primary focus has been on handling simple undirected graphs. Multi-relational graphs are a more general and prevalent form of graphs where each edge has a label and direction associated with it. Most of the existing approaches to handle such graphs suffer from over-parameterization and are restricted to learning representations of nodes only. In this paper, the authors propose CompGCN, a novel Graph Convolutional framework which jointly embeds both nodes and relations in a relational graph. CompGCN leverages a variety of entity-relation composition operations from Knowledge Graph Embedding techniques and scales with the number of relations. It also generalizes several of the existing multi-relational GCN methods. The authors evaluate the proposed method on multiple tasks such as node classification, link prediction, and graph classification, and achieve demonstrably superior results. The authors make the source code of CompGCN available to foster reproducible research.

- Paper
[Composition-based Multi-Relational Graph Convolutional Networks](https://arxiv.org/abs/1911.03082) 

- 模型代码来源
  
This repository includes software from https://github.com/dmlc/dgl/tree/master/examples/pytorch/compGCN
licensed under the Apache License, Version 2.0.

Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.

### 数据集
#### 数据集下载地址
  - wn18rr数据集(WordNet-18 RR)
  下载地址：https://dgl-data.s3.cn-north-1.amazonaws.com.cn/dataset/wn18rr.zip

  - FB15k-237数据集
  下载地址：https://dgl-data.s3.cn-north-1.amazonaws.com.cn/dataset/FB15k-237.zip



#### 预处理
- FB15k-237数据集
unzip FB15k-237.zip

- wn18rr数据集
unzip wn18rr.zip

### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- |
| Nvidia GPU | ✅       |
| 昆仑芯 XPU | N/A     |
| 天数智芯   | N/A     |
