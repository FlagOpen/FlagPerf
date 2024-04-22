### 1. 推理数据集

* 下载地址：`https://huggingface.co/datasets/Stevross/mmlu/tree/main`
  1. 下载其中的data.tar
  2. 将.tar文件还原为目录
  3. 将解压后的data目录放置在config.data_dir/config.mmlu_dir

### 2. 模型与权重

* 模型实现
  * pytorch：transformers.AutoModelForCausalLM
* 权重加载
  * pytorch：AutoModelForCausalLM.from_pretrained(config.data_dir/config.weight_dir)
* 权重获取方式
  1. 下载`https://hf-mirror.com/deepseek-ai/deepseek-llm-7b-base/tree/main`目录下的全部文件
  2. 将下载好的权重保存至config.data_dir/config.weight_dir

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
| inductor | fp16      | 2305     | 8877.0      | 8913.4     | 9383.8        | 10338.8        | 46.4% | 48.2%/48.2% | 28.7/40.0 |
