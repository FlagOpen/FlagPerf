### 模型信息
- 模型介绍
>中文预训练语言模型（CPM）是基于transformers 的自回归语言模型，其训练使用了100G中文数据，最大版本包含26亿参数，支持文本分类、文本生成。 
>获取CPM论文了解更多 
>[CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)

- 模型代码来源

| repo    | commmit_id  | date |
|  ----  | ----  |----  |
| [CPM-1-Finetune](https://github.com/TsinghuaAI/CPM-1-Finetune) | c0d892185912b28f8efeaeb55905f3f4fb227e46|2021-10-17 21:53:00|

### 模型Checkpoint下载
> [下载页](https://model.baai.ac.cn/model-detail/100017)
文件及版本tab页下，pytorch_model.bin. 
参数数：2.6B

### 测试数据集下载
> Dataset : https://drive.google.com/drive/folders/1gL01xbFBcrgP0TmgOhJ_uplkeG-BCwvM

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
| 训练资源 | 配置文件            | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s) |
| -------- | ---------------   | ---------- | ------- | -------  | ------- | --------------- |
| 单机1卡  | config_BI-V100x1x1 | pending    | 0.8     | pending  | pending |pending          |
| 单机2卡  | config_BI-V100x1x2 | pending    | 0.8     | pending  | pending |pending          |
| 单机4卡  | config_BI-V100x1x4 | pending    | 0.8     | pending  | pending |pending          |
| 单机8卡  | config_BI-V100x1x8 | pending    | 0.92    | pending  | pending |pending          |
| 两机8卡  | config_BI-V100x2x8 | pending    | 0.92    | pending  | pending |pending          |

### 许可证

本项目基于Apache 2.0 license。

本项目部分代码基于MLCommons https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA 实现。