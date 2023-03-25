### 模型信息
- 模型介绍
>GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.
>Please refer to our paper for a detailed description of GLM:
>[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (ACL 2022)

- 模型代码来源
> https://github.com/THUDM/GLM

### 模型Checkpoint下载
> `https://cloud.tsinghua.edu.cn/d/13f5b03da9594e5490c4/files/?p=%2Fglm-large-blank.tar.bz2`

### 测试数据集下载
> `https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip`

- 预处理
> 无需预处理 

### 天数智芯 BI-V100 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本:  4.15.0-156-generic x86_64    
   - 加速卡驱动版本：3.0.0
   - Docker 版本：20.10.8
   - 训练框架版本：torch-1.10.2+corex.3.0.0
   - 依赖软件版本：无


### 运行情况
| 训练资源 |       配置文件     | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s) |
| -------- | ------------------ | ----------- | -------- | -------- | ------- | --------------- |
| 单机8卡  | config_BI-V100x1x8 | 10120.93    | 0.8      | 0.8081   | 1500    | 9.82            |
| 两机8卡  | config_BI-V100x2x8 | coming      | 0.8      | coming   | coming  | coming          |

### 许可证


本项目基于Apache 2.0 license。

本项目部分代码基于MLCommons https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA 实现。
