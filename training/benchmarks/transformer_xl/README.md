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

[WMT14](http://statmt.org/wmt14/translation-task.html#Download)

- 数据集下载和预处理

```
bash scripts/run_preprocessing.sh
```


## 3. 框架与芯片支持情况说明

- 目前FlagPerf提供 &lt;Framework&gt; 的实现.
- 目前已适配本模型的芯片如下：

|              | *Pytorch* | *Paddle* | *TensorFlow2* |
| ------------ | --------- | -------- | ------------- |
| *Nvidia GPU* |    *✅*   | *N/A*    | *N/A*         |
| *Kunlunxin XPU* | *✅*  | *N/A*    | *N/A*         |
