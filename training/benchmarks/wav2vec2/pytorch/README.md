### 模型信息

This repository provides an optimized implementation of the wav2vec 2.0 model, as described in the paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf). It is based on the [Fairseq codebase](https://github.com/facebookresearch/fairseq) published by the authors of the paper. The wav2vec 2.0 model is pre-trained unsupervised on large corpora of speech recordings. Afterward, it can be quickly fine-tuned in a supervised way for speech recognition or serve as an extractor of high-level features and pseudo-phonemes for other applications.

### 代码来源

https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/wav2vec2/


### 数据集下载地址(global proxy)
http://www.openslr.org/resources/12


### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- | 
| Nvidia GPU | ✅      | 
| 昆仑芯 XPU | N/A      |  
| 天数智芯   | N/A      |
