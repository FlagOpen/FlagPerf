### 1. 数据集准备

[ZINC 250k](../../benchmarks/moflow/pytorch/README.md#dataset) 

### 2. Nvidia GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器型号: NVIDIA DGX A100(40G) 
    - 加速卡型号: NVIDIA_A100-SXM4-40GB
    - CPU型号: AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，200Gb/s
    
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-1.13.0a0+936e930
   - 依赖软件版本：
     - cuda: 11.4
   

### 3.运行情况

* 通用指标

| 指标名称       | 指标值                  | 特殊说明                                                    |
| -------------- | ----------------------- | ----------------------------------------------------------- |
| 任务类别       | DrugDiscovery           |                                                             |
| 模型           | MoFlow                  |                                                             |
| 数据集         | ZINC 250k               |                                                             |
| 数据精度       | precision,见“性能指标”  | 可选tf32/amp/fp16/                                          |
| 超参修改       | fix_hp,见“性能指标”     | 跑满硬件设备评测吞吐量所需特殊超参                          |
| 硬件设备简称   | nvidia A100             |                                                             |
| 硬件存储使用   | mem,见“性能指标”        | 通常称为“显存”,单位为GiB                                    |
| 端到端时间     | e2e_time,见“性能指标”   | 总时间+Perf初始化等时间                                     |
| 总吞吐量       | p_whole,见“性能指标”    | 实际训练图片数除以总时间(performance_whole)                 |
| 训练吞吐量     | p_train,见“性能指标”    | 不包含每个epoch末尾的评估部分耗时                           |
| **计算吞吐量** | **p_core,见“性能指标”** | 不包含数据IO部分的耗时(p_core>p_train>p_whole)              |
| 训练结果       | nuv,见“性能指标”        | 所有生成的分子中，Novel, Unique and Valid的分子所占的百分比 |
| 额外修改项     | 无                      |                                                             |

* 性能指标

| 配置              | precision | fix_hp            | e2e_time | p_whole | p_train | p_core | final_nuv | mem       |
| ----------------- | --------- | ----------------- | -------- | ------- | ------- | ------ | --------- | --------- |
| A100单机8卡(1x8)  | amp       | /                 | 3220     | 27023   | 27519   | 30789  | 88.45     | 11.8/40.0 |
| A100单机8卡(1x8)  | amp       | bs=3072,lr=0.003  | /        | 31279   | 38043   | 46823  | /         | 34.6/40.0 |
| A100单机单卡(1x1) | amp       | bs=3584,lr=0.0001 | /        | 5810    | 5992    | 6387   | /         | 37.9/40.0 |
| A100两机8卡(2x8)  | amp       | bs=3072,lr=0.0005 | /        | 47655   | 63957   | 90228  | /         | 34.5/40.0 |



> 注
> 原始仓库中的[NUV](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/DrugDiscovery/MoFlow#results)取了**20**个不同的随机种子，进行了**20**次实验的平均值。
> 此模型本身的实验结果对随机性比较敏感。seed, temperature等都会影响nuv的值，参考[training-stability-test]一节的说明。(https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/DrugDiscovery/MoFlow#training-stability-test)。
> 如厂商一次无法收敛，需尝试运行**若干次**。

