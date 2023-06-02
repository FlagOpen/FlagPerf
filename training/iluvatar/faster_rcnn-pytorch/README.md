### 模型backbone权重下载
[模型backbone权重下载](../../benchmarks/faster_rcnn) 

这一部分路径在FlagPerf/training/benchmarks/faster_rcnn/pytorch/model/\_\_init__.py中提供：

```python
torchvision.models.resnet.__dict__['model_urls'][
    'resnet50'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
```
本case中默认配置为，从官网同路径（0676ba61）自动下载backbone权重。用户如需手动指定，可自行下载至被挂载到容器内的路径下，并于此处修改路径为"file://"+download_path

### 测试数据集下载

[测试数据集下载](../../benchmarks/faster_rcnn) 

### 天数智芯 BI-V100 GPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
    - 机器、加速卡型号: Iluvatar BI-V100 32GB

- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本:  4.15.0-156-generic x86_64    
   - 加速卡驱动版本：3.0.0
   - Docker 版本：20.10.8
   - 训练框架版本：torch-1.10.2+corex.3.0.0
   - 依赖软件版本：无


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(mAP) | 性能（samples/s） |
| -------- | --------------- | ----------- | -------- | ------------- | ----------------- |
| 单机8卡  | config_BI-Vx1x8 |     | 0.35     | 0.3654       |             |

