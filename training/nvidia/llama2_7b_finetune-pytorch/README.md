### Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器型号:  DGXA100
    - 加速卡型号:  NVIDIA A100-SXM4-40GB
    - CPU型号: AMD EPYC 7742 64-Core Processor
    - 多机网络类型、带宽: InfiniBand，200Gb/s
    
- ##### 软件环境
   - OS版本：Ubuntu 20.04.4
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 依赖软件版本：见llama2_7b_finetune-pytorch/config/requirements.txt

- ##### 并行策略

   - 并行技术：无
   - 实施者：无
   - 实施细节：无

- ##### 优化策略

   - 优化方案：lora
   - 方案细节：LoraConfig(
      auto_mapping=None, 
      base_model_name_or_path=None, 
      revision=None, task_type='CAUSAL_LM', 
      inference_mode=False, r=8, 
      target_modules=['q_proj', 'v_proj'], 
      lora_alpha=32, lora_dropout=0.05, 
      fan_in_fan_out=False, bias='none', 
      modules_to_save=None, 
      init_lora_weights=True, 
      layers_to_transform=None, 
      layers_pattern=None)

### 运行情况

* 输入批尺寸
  1. local_batchsize(batch_size_training)，简写为LBS，即实际进入模型的张量批尺寸，为config_A100x1x1.py中所写，在本case中默认为2
  2. seq_length(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，为config_A100x1x1.py中所写，在本case中默认为512
  3. gradient_accumulate_steps，简写为GAS，即梯度累加步数，为ds_config.json中所写，在本case中默认为1
  4. global_batchsize恒等于local_batchsize*world_size，本case单卡运行因此为2.

* 通用指标

| 指标名称     | 指标值                     | 特殊说明                           |
| ------------ | -------------------------- | ---------------------------------- |
| 任务类别     | 自然语言理解               |                                    |
| 模型         | llama2_7b                  |                                    |
| 数据集       | openwebtext                | 如无特殊说明，训练前1亿个token |
| 数据精度     |fp32                        |                                    |
| 超参修改     | fix_hp,见“性能指标”        | 运行必要特殊超参，例如需要改小seqlength避免OOM |
| 硬件设备简称 | nvidia A100                |                                    |
| 硬件存储使用 | mem,见“性能指标”           | 通常称为“显存”,单位为GiB           |
| 计算使用率 | MFU,见“性能指标”           | 参见PaLM论文定义 |
| **吞吐量**   | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数          |
| MMLU结果           | acc(推理/验证)   | MMLU回答准确率（few_shots:5）                   |
* 性能指标

| 配置                |  fix_hp           | token/p/s | loss | mem       |acc(MMLU) |MFU       |
| ------------------- | ---------------- | ------ | ------- | --------- | --------- |--------- |
| A100单机单卡（1x1）  |  /  | 2788 | 1.64 | 37.3/40 | 0.38 |/|
| A100单机单卡（1x1）  |  数据精度=fp16, local_batchsize=4  | 4017 | 1.77 | 32.0/40 | 0.43 |36.1%|

>注：
>finetune训练数据集为samsum_dataset,MMLU数据集在这里只做配合lora-finetune后功能测试使用，MMLU评测结果无finetune结果指导意义，这里关注吞吐量即可。