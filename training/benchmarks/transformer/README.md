## 1. 模型信息

- 模型介绍

Transformer是一种神经机器翻译（NMT）模型，它使用注意力机制来提高训练速度和整体准确性。Transformer模型最初在[Attention Is All You Need](https://arxiv.org/abs/1706.03762)中被介绍，并在[Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187)中得到了改进。此实现基于Facebook建立在PyTorch之上的Fairseq NLP工具包中的优化实现。

- 论文

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187)

- 模型代码来源

This case includes code from the BSD License open source project at https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.


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
