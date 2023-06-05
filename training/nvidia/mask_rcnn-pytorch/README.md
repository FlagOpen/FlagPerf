### 模型Checkpoint下载
[模型Checkpoint下载](../../benchmarks/mask_rcnn/README.md#模型checkpoint)
### 测试数据集下载
[测试数据集下载](../../benchmarks/mask_rcnn/README.md#数据集下载地址)

### Nvidia GPU配置与运行信息参考
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
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标mAP精度(bbox && segm) | 收敛mAP精度(bbox && segm) | 性能(samples/s) |
| -------- | --------------- | ----------- | ------------------------- | ------------------------- | --------------- |
| 单机8卡  | config_A100x1x8 | 14806.69    | 0.38 && 0.34              | 0.383 && 0.344            | 143.39          |

精度说明：torchvision的 MaskRCNN 的ResNet50 backbone，在batch_size=2, lr=0.02时，精度为map_bbox=0.379, map_segm=0.346.
FlagPerf的精度为了达到或者接近原始模型的精度，并且尽可能提高单卡GPU占用率，增大了batch_size和lr, 训练的精度和原始模型相比，略有差异。
增大batch_size的同时，适当提高lr，会提升训练精度map，参见PR附件训练日志。
