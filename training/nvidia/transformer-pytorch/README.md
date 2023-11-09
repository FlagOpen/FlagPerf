### 测试数据集下载

[测试数据集下载](../../benchmarks/transformer/README.md#数据集)

### Nvidia GPU配置与运行信息参考

#### 环境配置

- ##### 硬件环境
  - 机器型号: NVIDIA DGX A100(40G) 
  - 加速卡型号: NVIDIA_A100-SXM4-40GB
  - CPU型号: AMD EPYC7742-64core@1.5G
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境

  - OS版本：Ubuntu 18.04
  - OS kernel版本: 5.4.0-113-generic
  - 加速卡驱动版本：470.129.06
  - Docker 版本：20.10.16
  - 训练框架版本：1.12.1+cu113
  - 依赖软件版本：无


### 运行情况

* 通用指标

| 指标名称       | 指标值                    | 特殊说明                                                                                                                                                      |
| -------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 任务类别       | Language Modelling && LLM |                                                                                                                                                               |
| 模型           | Transformer               |                                                                                                                                                               |
| 数据集         | WMT14                     | http://statmt.org/wmt14/translation-task.html#Download                                                                                                        |
| 数据精度       | precision,见“性能指标”    | 可选fp32/amp/fp16/tf32                                                                                                                                        |
| 超参修改       | fix_hp,见“性能指标”       | 跑满硬件设备评测吞吐量所需特殊超参                                                                                                                            |
| 硬件设备简称   | nvidia A100               |                                                                                                                                                               |
| 硬件存储使用   | mem,见“性能指标”          | 通常称为“显存”,单位为GiB                                                                                                                                      |
| 端到端时间     | e2e_time,见“性能指标”     | 总时间+Perf初始化等时间                                                                                                                                       |
| 总吞吐量       | p_whole,见“性能指标”      | 实际训练样本数除以总时间(performance_whole)                                                                                                                   |
| 训练吞吐量     | p_train,见“性能指标”      | 不包含每个epoch末尾的评估部分耗时                                                                                                                             |
| **计算吞吐量** | **p_core,见“性能指标”**   | 不包含数据IO部分的耗时(p3>p2>p1)                                                                                                                              |
| 训练结果       | bleu,见“性能指标”         | BLEU (BiLingual Evaluation Understudy) 是一种自动评估机器翻译文本的指标。 BLEU 得分是一个 0到1 之间的数字，用于衡量机器翻译文本与一组高质量参考翻译的相似度。 |
| 额外修改项     | 无                        |                                                                                                                                                               |

* 性能指标

| 配置              | precision | fix_hp | e2e_time | p_whole | p_train | p_core | final_bleu | mem       |
| ----------------- | --------- | ------ | -------- | ------- | ------- | ------ | ---------- | --------- |
| A100单机8卡(1x8)  | fp32      | /      | 4337     | 296248  | 324360  | 328478 | 27.08      | 31.2/40.0 |
| A100单机单卡(1x1) | fp32      | /      |          | 46999   | 47789   | 48398  |            | 32.6/40.0 |
| A100两机8卡(2x8)  | fp32      | /      |          | 520689  | 582954  | 589793 |            | 37.3/40.0 |


[官方精度](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#training-performance-nvidia-dgx-a100-8x-a100-40gb)为27.92，按照[官方配置](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#training-performance-nvidia-dgx-a100-8x-a100-40gb)，训完得到的精度为27.08
