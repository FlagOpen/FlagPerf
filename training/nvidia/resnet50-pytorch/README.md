### 1. 数据集准备
[下载ImageNet2012](../../benchmarks/resnet50) 

### 2. 模型Checkpoint下载
在xxx链接下载第70个checkpoint, 并在FlagPerf/training/benchmarks/resnet50/pytorch/config/_base.py 中init_checkpoint 指定路径, 即可从第70个epoch开始继续训练。

如果想要从头开始训练，以config_A100x1x8.py为例，设置配置文件中 init_checkpoint = None 即可。



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
   - 训练框架版本：pytorch-1.8.0a0+52ea372
   - 依赖软件版本：
     - cuda: 11.4


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s) |
| -------- | --------------- | ----------- | -------- | -------- | ------- | --------------- |
| 单机2卡  | config_A100x1x2 | 9318.51     | 0.760    | 0.7601   | 6000    | 330.03          |
| 单机4卡  | config_A100x1x4 | 1673.53     | 0.760    | 0.7601   | 1020    | 629.12          |
| 单机8卡  | config_A100x1x8 | 335.34      | 0.760    | 0.7601   | 180     | 1162.18         |
| 两机8卡  | config_A100x2x8 | 300.94      | 0.764    | 0.7643   | 165     | 2387.23         |
