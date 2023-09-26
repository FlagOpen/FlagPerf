### 1. 下载数据集和模型
[下载链接](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/datasets/distilbert_train.tar) 

### 2. 设置test_conf.py

为了使得`training/nvidia/distilbert-pytorch/config/requirements.txt`里的依赖库均能被下载，需要将`training/run_benchmarks/config/test_conf.py`里的`PIP_SOURCE`的值修改为`https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: InfiniBand，200Gb/s
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-1.12.0a0+bd13bc6
   - 依赖软件版本：
     - cuda: 11.6

### 运行情况

* 通用指标

| 指标名称       | 指标值                  | 特殊说明                              |
| -------------- | ----------------------- | ------------------------------------- |
| 任务类别       | Summarization                |                                       |
| 模型           | distilbert                |                                       |
| 数据集         | SST-2               |                                       |
| 超参修改       | fix_hp,见“性能指标” | 跑满硬件设备评测吞吐量所需特殊超参 |
| 硬件设备简称   | nvidia A100             |                                       |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB              |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间               |
| 总吞吐量       | p_whole,见“性能指标”    | 实际样本数数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时     |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1),单位为samples/s(seq_length=512)      |
| 训练结果       | acc,见“性能指标”        | 分类准确率            |
| 额外修改项     | 无                      |                                       |

* 性能指标

| 配置               | precision | fix_hp | e2e_time | p_whole | p_train | p_core | acc  | mem |
| ------------------ | --------- | ---- | ----      | ----     | ----   | ----   |  ---- |  ---- |
| A100单机8卡（1x8） | fp32        | /    | 361      | 1764.0    | 1861.9 | 1942.6 |  0.915 | 13.9 /40.0 |
