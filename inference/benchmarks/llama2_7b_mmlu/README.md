### 1. 推理数据集

* 下载地址：`https://huggingface.co/datasets/Stevross/mmlu/tree/main`
  1. 下载其中的data.tar
  2. 将.tar文件还原为目录
  3. 将解压后的data目录放置在config.data_dir/config.mmlu_dir

### 2. 模型与权重

* 模型实现
  * pytorch：transformers.LlamaForCausalLM
* 权重加载
  * pytorch：LlamaForCausalLM.from_pretrained(config.data_dir/config.weight_dir)
* 权重获取方式
  1. 填写申请表，向meta ai申请获取llama2模型权重，并同意相关协议
  2. 下载其中的llama2-7b权重（注意不是chat）
  3. 使用huggingface提供的convert.py将权重转化为huggingface格式，并保存在config.data_dir/config.weight_dir

### 3. 软硬件配置与运行信息参考

#### 3.1 Nvidia A100

- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04
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

### 4. 运行情况（Llama2_7b_MMLU）

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
| inductor | fp16      | 2558     | 8596.9      | 8630.3     | 9230.8        | 10052.2        | 45.1% | 45.8%/45.8% | 28.0/40.0 |
| inductor | fp32   | 4143     | 5455.3      | 5469.4     | 5675.7        | 5951.8         | 53.4% | 45.8%/45.8% | 35.0/40.0 |
