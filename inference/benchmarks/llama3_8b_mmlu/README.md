### 0. 模型背景

* 基本信息
  * Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

* 模型结构
  * Llama 3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

* Meta提供的llama3-8b-mmlu得分: Acc=65.4%
  * Related link: `https://github.com/meta-llama/llama3/blob/main/eval_details.md`

* 源代码链接
  * `https://github.com/meta-llama/llama3`

### 1. 推理数据集

* 下载地址：`https://huggingface.co/datasets/Stevross/mmlu/tree/main`
  1. 下载其中的data.tar
  2. 将.tar文件还原为目录
  3. 在config.data_dir目录下创建"mmlu_dataset"目录，将解压后的data目录放置在config.data_dir/mmludataset/目录下
  * config.data_dir为本case存放模型权重与推理数据集的目录，默认为"/raid/dataset/llama3_8b_mmlu"，如需更改，请修改Flagperf/inference/configs/host.yaml中CASES变量的value。
  * config.mmlu_dir为本case存放推理数据集的目录，默认为"/mmlu_dataset/data"，如需更改，请修改FlagPerf/inference/configs/llama3_8b_mmlu/parameters.yaml中的mmlu_dir变量。

  llama3_8b_mmlu<br/>
  &emsp;&emsp;├── llama3_8b_hf<br/>
  &emsp;&emsp;└── mmlu_dataset<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;└── data<br/>


### 2. 模型与权重

* 模型实现
  * pytorch：transformers.AutoModelForCausalLM
* 权重加载
  * pytorch：AutoModelForCausalLM.from_pretrained(config.data_dir/config.weight_dir)
* 权重获取方式
  1. 权重申请地址：`https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main`
  2. 填写申请表并同意相关协议，向meta ai申请llama3-8b权重
  3. 使用huggingface提供的convert.py将权重转换为huggingface格式，并保存在config.data_dir/config.weight_dir
  
  * config.data_dir为本case存放模型权重与推理数据集的目录，默认为"/raid/dataset/llama3_8b_mmlu"，如需更改，请修改Flagperf/inference/configs/host.yaml中CASES变量的value。
  * config.weight_dir为本case存放模型权重的目录，默认为"llama3_8b_hf"，如需更改，请修改FlagPerf/inference/configs/llama3_8b_mmlu/parameters.yaml中的weight_dir变量。

  llama3_8b_mmlu<br/>
  &emsp;&emsp;├── llama3_8b_hf<br/>
  &emsp;&emsp;└── mmlu_dataset<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;└── data<br/>

### 3. 软硬件配置与运行信息参考

#### 3.1 Nvidia A100

- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: 此样例不涉及服务器间通信

- ##### 软件环境
   - OS版本：Ubuntu 20.04.4 LTS
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-2.1.0a0+4136153
   - 依赖软件版本：
     - cuda: 12.1
   
   - 推理工具包
   - Inductor (torch._dynamo) pytorch-2.1.0a0+4136153
   
- ##### 优化策略

   - None

- ##### 并行策略

   - None



### 4. 运行情况（deepseek_7b_MMLU）

* 指标列表

| 指标名称           | 指标值索引        | 特殊说明                                                    |
| ------------------ | ----------------- | ----------------------------------------------------------- |
| 数据精度           | precision         | 可选fp32/fp16                                               |
| 硬件存储使用       | mem               | 通常称为“显存”,单位为GiB                                    |
| 端到端时间         | e2e_time          | 总时间+Perf初始化等时间                                     |
| 验证总吞吐量       | p_val_whole       | 实际验证序列数除以总验证时间                                |
| 验证计算吞吐量     | p_val_core       | 不包含IO部分耗时                                            |
| 推理总吞吐量       | p_infer_whole     | 实际推理序列数除以总推理时间                                |
| **推理计算吞吐量** | **\*p_infer_core** | 不包含IO部分耗时                             |
| **计算卡使用率** | **\*MFU** | model flops utilization                             |
| 推理结果           | acc(推理/验证)    | 单位为5-shots MMLU回答准确率                            |

* 指标值


| 推理工具  | precision | e2e_time | p_val_whole | p_val_core | p_infer_whole | \*p_infer_core | \*MFU     | acc         | mem        |
| ----------- | --------- | ---- | ---- | -------- | ----------- | ---------- | ------------- | ------------ | ----------- |
| inductor | fp16      | 2415     | 7945.9      | 7979.6     | 8178.3        | 9051.3        | 46.4% | 65.2%/65.2% | 34.5/40.0 |
