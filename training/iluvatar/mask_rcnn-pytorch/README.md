### 模型Checkpoint下载
[模型Checkpoint下载](../../benchmarks/mask_rcnn/README.md#模型checkpoint)
### 测试数据集下载
[测试数据集下载](../../benchmarks/mask_rcnn/README.md#数据集下载地址)

### 天数智芯 BI-V100 GPU配置与运行信息参考
#### 环境配置

- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 18.04
   - OS kernel版本:  5.4.0-150-generic x86_64    
   - 加速卡驱动版本：3.1.0
   - Docker 版本：24.0.2
   - 训练框架版本：torch-1.13.1+corex.3.1.0
   - 依赖软件版本：无

#### 训练时遇到从pytorch官网下载resnet权重比较慢的情况，可以手动拷贝本地的resnet50.pth指定路径，参考[environment_variables.sh](config/environment_variables.sh#environment_variables.sh)

### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标mAP精度(bbox && segm) | 收敛mAP精度(bbox && segm) | 性能(samples/s) |
| -------- | --------------- | ----------- | ------------------------- | ------------------------- | --------------- |
| 单机8卡  | config_BI-V100x1x8 |     | 0.38 && 0.34              | 0.384 && 0.346            |          |
