### 模型信息

This repository provides an optimized implementation of the wav2vec 2.0 model, as described in the paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf). It is based on the [Fairseq codebase](https://github.com/facebookresearch/fairseq) published by the authors of the paper. The wav2vec 2.0 model is pre-trained unsupervised on large corpora of speech recordings. Afterward, it can be quickly fine-tuned in a supervised way for speech recognition or serve as an extractor of high-level features and pseudo-phonemes for other applications.

### 代码来源

This case includes code from the BSD 3-Clause License open source project at https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/wav2vec2/

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.


### 数据集下载地址(global proxy)
数据来源 https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/wav2vec2#quick-start-guide

执行

DATASET_DIR=[PATH]  bash training/benchmarks/wav2vec2/pytorch/scripts/download_data.sh

DATASET_DIR=[PATH]  bash training/benchmarks/wav2vec2/pytorch/scripts/generate_filelists.sh

### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- | 
| Nvidia GPU | ✅      | 
| 昆仑芯 XPU | N/A      |  
| 天数智芯   | N/A      |
