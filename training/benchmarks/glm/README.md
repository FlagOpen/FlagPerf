### 模型信息
- 模型介绍
>GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.
>Please refer to our paper for a detailed description of GLM:
>[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (ACL 2022)

- 模型代码来源

This case includes code from the MIT License open source project at https://github.com/THUDM/GLM

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


### 数据集
- 数据集及checkpoint下载地址
>`https://model.baai.ac.cn/model-detail/100097`  
> 文件名：`glm_train_dataset.zip`

- 预处理
- 无需预处理，解压缩数据集即可。
```bash
unzip glm_train_dataset.zip
```
解压后的目录结构
```bash
├── ReCoRD
│   └── glm_train_eval_hdf5_sparse
│       ├── eval_hdf5
│       │   └── eval_sparse.hdf5
│       └── train_hdf5
│           └── train_sparse.hdf5
├── blocklm-large-blank
│   ├── 200000
│   │   └── mp_rank_00_model_states.pt
│   └── latest_checkpointed_iteration.txt
```




### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | ✅ |N/A  |N/A|
| 天数智芯 GPU | ✅ |N/A  |N/A|



