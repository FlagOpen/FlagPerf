### 模型信息
#### 模型介绍
中文预训练语言模型（CPM）是基于transformers 的自回归语言模型，其训练使用了100G中文数据，最大版本包含26亿参数，支持文本分类、文本生成。 
获取CPM论文了解更多 
[CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)

#### 模型代码来源
| repo    | commmit_id  | date |
|  ----  | ----  |----  |
| [CPM-1-Finetune](https://github.com/TsinghuaAI/CPM-1-Finetune) | c0d892185912b28f8efeaeb55905f3f4fb227e46|2021-10-17 21:53:00|

This case includes code from the MIT License open source project at https://github.com/TsinghuaAI/CPM-1-Finetune

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.



#### 数据集
##### 测试数据集下载地址
[Test Dataset](https://drive.google.com/drive/folders/1gL01xbFBcrgP0TmgOhJ_uplkeG-BCwvM)

##### 预处理
> 无需预处理 

#### 模型checkpoint 
[下载页](https://model.baai.ac.cn/model-detail/100105)
文件及版本tab页下，cpm_model_states_medium.pt
参数数：0.33B

### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 天数智芯   |  ✅  | N/A |N/A|


