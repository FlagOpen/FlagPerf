## 1. 模型信息

- 模型介绍

The Transformer-XL model was proposed in [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov. It’s a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can reuse previously computed hidden-states to attend to longer context (memory). This model also uses adaptive softmax inputs and outputs (tied).

- 论文

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

- 模型代码来源

Pytorch case:
 This repository includes software from https://github.com/huggingface/transformers/tree/v4.33.0
 licensed under the Apache License 2.0.

 Some of the files in this directory were modified by BAAI in 2023 to support FlagPerf.


## 2. 数据集

https://paperswithcode.com/dataset/wikitext-103

The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License.

The dataset is available on huggingface: https://huggingface.co/datasets/wikitext. And tokenizer is available at https://huggingface.co/transfo-xl-wt103.

The dataset should be organized as follow

```
data_dir/
├── data
│   ├── LICENSE
│   ├── dataset_info.json
│   ├── wikitext-test.arrow
│   ├── wikitext-train.arrow
│   └── wikitext-validation.arrow
└── model
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.bin
    └── vocab.pkl
```


## 3. 框架与芯片支持情况说明

- 目前FlagPerf提供 &lt;Framework&gt; 的实现.
- 目前已适配本模型的芯片如下：

|              | *Pytorch* | *Paddle* | *TensorFlow2* |
| ------------ | --------- | -------- | ------------- |
| *Nvidia GPU* |    *✅*   | *N/A*    | *N/A*         |
| *Kunlunxin XPU* | *N/A*  | *N/A*    | *N/A*         |
