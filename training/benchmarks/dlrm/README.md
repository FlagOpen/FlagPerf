### 模型信息
- Introduction

With the advent of deep learning, neural network-based recommendation models have emerged as an important tool for tackling personalization and recommendation tasks. These networks differ significantly from other deep learning networks due to their need to handle categorical features and are not well studied or understood. In this paper, the authors develop a state-of-the-art deep learning recommendation model (DLRM) and provide its implementation in both PyTorch and Caffe2 frameworks. In addition, the authors design a specialized parallelization scheme utilizing model parallelism on the embedding tables to mitigate memory constraints while exploiting data parallelism to scale-out compute from the fully-connected layers. The authors compare DLRM against existing recommendation models and characterize its performance on the Big Basin AI platform, demonstrating its usefulness as a benchmark for future algorithmic experimentation and system co-design.

- Paper
[Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091) 

- 模型代码来源
This case includes code from the BSD 3-Clause License open source project at https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

### 数据集
#### 数据集下载地址
  Criteo Terabyte Dataset.
  Website：https://labs.criteo.com/2013/12/download-terabyte-click-logs/


#### 预处理


1. Clone the repository.
   ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/Recommendation/DLRM
   ```
2. Download the dataset.
  You can download the data by following the instructions at: http://labs.criteo.com/2013/12/download-terabyte-click-logs/. When you have successfully downloaded it and unpacked it, set the CRITEO_DATASET_PARENT_DIRECTORY to its parent directory:

  ```bash
  CRITEO_DATASET_PARENT_DIRECTORY=/raid/dataset/criteo_1TB_click_logs
  ```
  We recommend to choose the fastest possible file system, otherwise it may lead to an IO bottleneck.


3. Build DLRM Docker containers
   
   ```bash
   docker build -t nvidia_dlrm_pyt --name=nvidia_dlrm_pytorch .
   docker build -t nvidia_dlrm_preprocessing --name=nvidia_dlrm_preprocessing -f Dockerfile_preprocessing . --build-arg DGX_VERSION=[DGX-2|DGX-A100]
   ```

  then, start an interactive session in the NGC container to run preprocessing. The DLRM PyTorch container can be launched with:

  ```bash
  docker run --runtime=nvidia -it --rm --ipc=host -v ${CRITEO_DATASET_PARENT_DIRECTORY}:/data/dlrm  --name=nvidia_dlrm_preprocessing nvidia_dlrm_preprocessing bash
  ```  

   
4. Preprocess the dataset.
   
  Here are a few examples of different preprocessing commands. Out of the box, we support preprocessing on DGX-2 and DGX A100 systems. For the details on how those scripts work and detailed description of dataset types (small FL=15, large FL=3, xlarge FL=2), system requirements, setup instructions for different systems and all the parameters consult the preprocessing section. For an explanation of the FL parameter, see the Dataset Guidelines and Preprocessing sections.

  Depending on dataset type (small FL=15, large FL=3, xlarge FL=2) run one of following command:

  4.1 Preprocess to small dataset (FL=15) with Spark GPU:

  ```bash
  cd /workspace/dlrm/preproc
  ./prepare_dataset.sh 15 GPU Spark
  ```

  prepare_dataset.sh will generate binary dataset in your CRITEO_DATASET_PARENT_DIRECTORY. you can preview the dataset using 
  
  ```bash
  cd ${CRITEO_DATASET_PARENT_DIRECTORY} && tree binary_dataset
  ```
  
  The following shows the binary_dataset structures.

  ```bash
  binary_dataset
├── feature_spec.yaml
├── test
│   ├── cat_0.bin
│   ├── cat_1.bin
│   ├── cat_10.bin
│   ├── cat_11.bin
│   ├── cat_12.bin
│   ├── cat_13.bin
│   ├── cat_14.bin
│   ├── cat_15.bin
│   ├── cat_16.bin
│   ├── cat_17.bin
│   ├── cat_18.bin
│   ├── cat_19.bin
│   ├── cat_2.bin
│   ├── cat_20.bin
│   ├── cat_21.bin
│   ├── cat_22.bin
│   ├── cat_23.bin
│   ├── cat_24.bin
│   ├── cat_25.bin
│   ├── cat_3.bin
│   ├── cat_4.bin
│   ├── cat_5.bin
│   ├── cat_6.bin
│   ├── cat_7.bin
│   ├── cat_8.bin
│   ├── cat_9.bin
│   ├── label.bin
│   └── numerical.bin
├── train
│   ├── cat_0.bin
│   ├── cat_1.bin
│   ├── cat_10.bin
│   ├── cat_11.bin
│   ├── cat_12.bin
│   ├── cat_13.bin
│   ├── cat_14.bin
│   ├── cat_15.bin
│   ├── cat_16.bin
│   ├── cat_17.bin
│   ├── cat_18.bin
│   ├── cat_19.bin
│   ├── cat_2.bin
│   ├── cat_20.bin
│   ├── cat_21.bin
│   ├── cat_22.bin
│   ├── cat_23.bin
│   ├── cat_24.bin
│   ├── cat_25.bin
│   ├── cat_3.bin
│   ├── cat_4.bin
│   ├── cat_5.bin
│   ├── cat_6.bin
│   ├── cat_7.bin
│   ├── cat_8.bin
│   ├── cat_9.bin
│   ├── label.bin
│   └── numerical.bin
└── validation
    ├── cat_0.bin
    ├── cat_1.bin
    ├── cat_10.bin
    ├── cat_11.bin
    ├── cat_12.bin
    ├── cat_13.bin
    ├── cat_14.bin
    ├── cat_15.bin
    ├── cat_16.bin
    ├── cat_17.bin
    ├── cat_18.bin
    ├── cat_19.bin
    ├── cat_2.bin
    ├── cat_20.bin
    ├── cat_21.bin
    ├── cat_22.bin
    ├── cat_23.bin
    ├── cat_24.bin
    ├── cat_25.bin
    ├── cat_3.bin
    ├── cat_4.bin
    ├── cat_5.bin
    ├── cat_6.bin
    ├── cat_7.bin
    ├── cat_8.bin
    ├── cat_9.bin
    ├── label.bin
    └── numerical.bin

3 directories, 85 files
  ```



### 框架与芯片支持情况
|            | Pytorch |
| ---------- | ------- |
| Nvidia GPU | ✅       |
| 昆仑芯 XPU | N/A     |
| 天数智芯   | N/A     |


