### 模型信息
- Introduction

The WaveGlow model is a flow-based generative model that generates audio samples from Gaussian distribution using mel-spectrogram conditioning (Figure 2). During training, the model learns to transform the dataset distribution into spherical Gaussian distribution through a series of flows. One step of a flow consists of an invertible convolution, followed by a modified WaveNet architecture that serves as an affine coupling layer. During inference, the network is inverted and audio samples are generated from the Gaussian distribution. Our implementation uses 512 residual channels in the coupling layer.

- Paper
[WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002) 

- 模型代码来源
This case includes code from the BSD3.0 protocol open source project at [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

### 数据集
#### 数据集下载地址
  The LJ Speech Dataset
  LJ Speech Dataset官网地址：https://keithito.com/LJ-Speech-Dataset/
  Dataset version: 1.1
  File md5sum: c4763be9595ddfa79c2fc6eaeb3b6c8e

  Statistics
  | Item                | Statistics |
  | ------------------- | ---------- |
  | Total Clips         | 13,100     |
  | Total Words         | 225,715    |
  | Total Characters    | 1,308,678  |
  | Total Duration      | 23:55:17   |
  | Mean Clip Duration  | 6.57 sec   |
  | Min Clip Duration   | 1.11 sec   |
  | Max Clip Duration   | 10.10 sec  |
  | Mean Words per Clip | 17.23      |
  | Distinct Words      | 13,821     |


#### 预处理
参考：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2

``` bash
.
├── LJSpeech-1.1
│   ├── README
│   ├── mels
│   ├── metadata.csv
│   └── wavs
└── filelists
    ├── ljs_audio_text_test_filelist.txt
    ├── ljs_audio_text_train_filelist.txt
    ├── ljs_audio_text_train_subset_1250_filelist.txt
    ├── ljs_audio_text_train_subset_2500_filelist.txt
    ├── ljs_audio_text_train_subset_300_filelist.txt
    ├── ljs_audio_text_train_subset_625_filelist.txt
    ├── ljs_audio_text_train_subset_64_filelist.txt
    ├── ljs_audio_text_val_filelist.txt
    ├── ljs_mel_text_filelist.txt
    ├── ljs_mel_text_test_filelist.txt
    ├── ljs_mel_text_train_filelist.txt
    ├── ljs_mel_text_train_subset_1250_filelist.txt
    ├── ljs_mel_text_train_subset_2500_filelist.txt
    ├── ljs_mel_text_train_subset_625_filelist.txt
    └── ljs_mel_text_val_filelist.txt
4 directories, 17 files
```


### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- |
| Nvidia GPU | ✅       |
| 昆仑芯 XPU | N/A     |
| 天数智芯   | N/A     |