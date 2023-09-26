### 1. 下载数据集和模型
[下载链接](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/datasets/t5_small_train.tar) 


### 2. 设置test_conf.py

为了使得`training/kunlunxin/t5_small-pytorch/config/requirements.txt`里的依赖库均能被下载，需要将`training/run_benchmarks/config/test_conf.py`里的`PIP_SOURCE`的值修改为`https://pypi.tuna.tsinghua.edu.cn/simple`


### 3. KUNLUNXIN XPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: 昆仑芯AI加速器组R480-X8
    - 加速卡型号: 昆仑芯AI加速卡R300
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本: 4.0.25
   - Docker 版本：20.10.16
   - 训练框架版本：xmlir+db597aa7
   - 依赖软件版本: pytorch-1.12.1+cpu

### 运行情况

* 通用指标

| 指标名称       | 指标值                  | 特殊说明                              |
| -------------- | ----------------------- | ------------------------------------- |
| 任务类别       | Summarization                |                                       |
| 模型           | t5_small                |                                       |
| 数据集         | CNN/Daily Mail            |                                       |
| 超参修改       | fix_hp,见“性能指标” | 跑满硬件设备评测吞吐量所需特殊超参 |
| 硬件设备简称   | kunlunxin R300             |                                       |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB              |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间               |
| 总吞吐量       | p_whole,见“性能指标”    | 实际样本数数除以总时间(performance_whole) |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时     |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p3>p2>p1)      |
| 训练结果       | rouge1,见“性能指标”        | rouge1分数            |
| 训练结果       | rouge2,见“性能指标”        | rouge2分数            |
| 训练结果       | rougeL,见“性能指标”        | rougeL分数            |
| 训练结果       | rougeLsum,见“性能指标”        | rougeLsum分数            |
| 额外修改项     | 无                      |                                       |

