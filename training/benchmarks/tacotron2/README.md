### 模型信息
- Introduction

  Tacotron 2 is a neural network architecture for speech synthesis directly from text. The system is composed of a recurrent sequence-to-sequence feature prediction network that maps character embeddings to mel-scale spectrograms, followed by a modified WaveNet model acting as a vocoder to synthesize timedomain waveforms from those spectrograms. The model achieves a mean opinion score (MOS) of 4.53 comparable to a MOS of 4.58 for professionally recorded speech. To validate the authors' design choices, the authors present ablation studies of key components of the system and evaluate the impact of using mel spectrograms as the input to WaveNet instead of linguistic, duration, and F0 features. The authors further demonstrate that using a compact acoustic intermediate representation enables significant simplification of the WaveNet architecture.

- Paper
[Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) 

- 模型代码来源
This case includes code from the BSD 3-Clause License open source project at https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2

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
1. 进入tacotron2数据集路径  cd  <YOUR_TACOTRON2_DATASET_PATH>
2. 下载数据集 wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
3. 解压缩  tar xvf LJSpeech-1.1.tar.bz2
4. git clone https://github.com/NVIDIA/DeepLearningExamples
5. 拷贝filelists目录到数据集路径。 cp -R DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/filelists <YOUR_TACOTRON2_DATASET_PATH>
6. 将tacotron2/scripts/prepare_mels.sh脚本拷贝到  <YOUR_TACOTRON2_DATASET_PATH>
7. 生成LJSpeech-1.1/mels目录下的数据：cd <YOUR_TACOTRON2_DATASET_PATH> && sh prepare_mels.sh
8. tree . -L 2，查看目录结构如下
  
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
| 天数智芯   |  ✅    |
