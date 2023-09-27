### 1. 推理数据集

* 采用单张图及其在原始huggingface doc中SamModel（cpu）上运行的结果作为推理数据集与评估指标
* 数据集下载
  * 在dataloader.py中自动下载
* groundTruth制作
  * 运行hugging face文档中使用SamModel推理的样例，将其计算出的3个image_size个bool mask矩阵按照mask从小到大的顺序依次存储为sam_gt_0.pt~sam_gt_2.pt，并放置在data_dir下

### 2. 模型与权重

* 模型实现
  * pytorch：transformers.SamModel
* 权重下载
  * pytorch：from_pretrained("facebook/sam_vit_huge")（hugging face）

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

  - TensorRT 8.6.1

#### 2.2 昆仑芯R200

- ##### 硬件环境
    - 机器、加速卡型号: R200

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.15.0-56-generic
   - 加速卡驱动版本：4.0
   - Docker 版本：20.10.21
   - 依赖软件版本：
     - pytorch: 1.13.0+cpu
     - onnx: 1.14.0

- 推理工具包
   
   - XTCL 2.0.0.67

### 3. 运行情况

* 指标列表

| 指标名称           | 指标值索引       | 特殊说明                                     |
| ------------------ | ---------------- | -------------------------------------------- |
| 数据精度           | precision        | 可选fp32/fp16                                |
| 批尺寸             | bs               |                                              |
| 硬件存储使用       | mem              | 通常称为“显存”,单位为GiB                     |
| 端到端时间         | e2e_time         | 总时间+Perf初始化等时间                      |
| 验证总吞吐量       | p_val_whole      | 实际验证图片数除以总验证时间                 |
| 验证计算吞吐量     | p_val_core       | 不包含IO部分耗时                             |
| 推理总吞吐量       | p_infer_whole    | 实际推理图片数除以总推理时间                 |
| **推理计算吞吐量** | **\*p_infer_core** | 不包含IO部分耗时                             |
| **计算卡使用率** | **\*MFU** | model flops utilization                             |
| 推理结果           | acc(推理/验证)   | 单位为top1分类准确率(acc1)                   |

* 指标值

| 推理工具  | precision | bs   | e2e_time | p_val_whole | p_val_core | p_infer_whole | \*p_infer_core | \*MFU     | acc         | mem        |
| ----------- | --------- | ---- | ---- | -------- | ----------- | ---------- | ------------- | ------------ | ----------- | ----------- |
| tensorrt | fp16    | 4   |1895.1 | 9.3 | 10.7 | 7.9 | 11.8 | 11.8% | 0.89/1.0 | 23.7/40.0 |
| tensorrt | fp32   | 2 | 1895.1 | 6.8 | 7.5 | 5.5         | 7.0 | 13.9% | 1.0/1.0 | 18.1/40.0 |

