### 1. 推理数据集
> Download website：https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl


### 2. 模型与权重

* 在data_dir中，共需要放置如下内容：

  ├── ckpts
  │   ├── iter_0135000
  │   │   ├── mp_rank_00
  │   │   │   └── model_optim_rng.pt
  │   │   ├── mp_rank_01
  │   │   │   └── model_optim_rng.pt
  │   │   ├── mp_rank_02
  │   │   │   └── model_optim_rng.pt
  │   │   ├── mp_rank_03
  │   │   │   └── model_optim_rng.pt
  │   │   ├── mp_rank_04
  │   │   │   └── model_optim_rng.pt
  │   │   ├── mp_rank_05
  │   │   │   └── model_optim_rng.pt
  │   │   ├── mp_rank_06
  │   │   │   └── model_optim_rng.pt
  │   │   └── mp_rank_07
  │   │       └── model_optim_rng.pt
  │   └── latest_checkpointed_iteration.txt
  ├── data
  │   └── lambada_test_bak.jsonl
  └── tokenizer
      ├── merges.txt
      ├── special_tokens.txt
      └── vocab.json

* 其中

  * lambada_test_bak.jsonl为推理数据集
  * tokenizer为aquila33B开源模型提供：https://model.baai.ac.cn/model-detail/100119
  * ckpts从上述路径https://model.baai.ac.cn/model-detail/100119获取所有.bin文件后，依照FlagScalehttps://github.com/FlagOpen/FlagScale描述方法，进行合并后重新划分为8个张量并行子权重

### 2. 软硬件配置与运行信息参考

#### 2.1 Nvidia A100


- ##### 并行策略

   - 张量并行，尺度为8

- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: InfiniBand，200Gb/s
    
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：1.12.0a0+8a1a93a
   - 依赖软件版本：
     - cuda: 11.8

- 推理工具包
  
   - FlagScale

### 3. 运行情况




* 指标列表

| 指标名称           | 指标值索引       | 特殊说明                                     |
| ------------------ | ---------------- | -------------------------------------------- |
| 数据精度           | precision        | 可选fp32/fp16                                |
| 硬件存储使用       | mem              | 通常称为“显存”,单位为GiB                     |
| 端到端时间         | e2e_time         | 总时间+Perf初始化等时间                      |
| 验证总吞吐量       | p_val_whole      | 实际验证token数除以总验证时间                 |
| 验证计算吞吐量     | p_val_core       | 不包含IO部分耗时                             |
| 推理总吞吐量       | p_infer_whole    | 实际推理token数除以总推理时间                 |
| **推理计算吞吐量** | **\*p_infer_core** | 不包含IO部分耗时                             |
| 推理结果           | loss | transformer-decoder loss（即平均压缩编码长度） |

* 指标值

| 推理工具  | precision | e2e_time | p_val_whole | \*p_val_core | p_infer_whole | \*p_infer_core |\*MFU| loss        | mem        |
| ----------- | --------- | -------- | ----------- | ---------- | ------------- | ------------ |  ------------ |----------- | ---------- |
| FlagScale | fp16   | 2479.9 | /           |  /  | 8711 | 8724 |23.8%| 3.0075 | 12.3/40.0 |
