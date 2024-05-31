
## Model Introduction
MoFlow is a model for molecule generation that leverages Normalizing Flows. Normalizing Flows is a class of generative neural networks that directly models the probability density of the data. They consist of a sequence of invertible transformations that convert the input data that follow some hard-to-model distribution into a latent code that follows a normal distribution which can then be easily used for sampling.

MoFlow was first introduced by Chengxi Zang et al. in their paper titled "MoFlow: An Invertible Flow Model for Generating Molecular Graphs" [paper](https://arxiv.org/pdf/2006.10137.pdf).



## Model source code
This repository includes software from [MoFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/DrugDiscovery/MoFlow)
licensed under the Apache License, Version 2.0 

Some of the files in this directory were modified by BAAI in 2024 to support FlagPerf.

## Dataset
### getting the data
This [original source code repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/DrugDiscovery/MoFlow#getting-the-data) contains the prepare_datasets.sh script that will automatically download and process the dataset. By default, data will be downloaded to the /data/ directory in the container.
```bash
bash prepare_datasets.sh
```
### preprocess the dataset
Start the container with [Dockerfile](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/DrugDiscovery/MoFlow/Dockerfile). Enter the container.
excute the folowing script to preprocess the dataset.
```bash
python3 scripts/data_preprocess.py
```

### dataset strucutres
preview directory structures of \<data_dir\>
```bash
tree .
```

```
.
├── valid_idx_zinc250k.json
├── zinc250k.csv
└── zinc250k_relgcn_kekulized_ggnp.npz
```

| FileName                           | Size(Bytes) | MD5                              |
| ---------------------------------- | ----------- | -------------------------------- |
| valid_idx_zinc250k.json            | 187832      | f8045b49a413c31136a0645d30c0b846 |
| zinc250k.csv                       | 23736231    | cd330eafb7a2cc413b3c9cafaf3efece |
| zinc250k_relgcn_kekulized_ggnp.npz | 375680462   | c91985e309a9f76457169859dbe1e662 |


## Checkpoint
- None

## AI Frameworks && Accelerators supports

|            | Pytorch                                    | Paddle | TensorFlow2 |
| ---------- | ------------------------------------------ | ------ | ----------- |
| Nvidia GPU | [✅](../../nvidia/moflow-pytorch/README.md) | N/A    | N/A         |
