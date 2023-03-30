# case README文件模版

> 文档信息说明：
> - 文档位置：每个Case的REAMDE文档位于training/benchmarks/&lt;model&gt;/&lt;Framework&gt; 目录下
> - 文档使用的语言：默认为中文README.md，可提供英文版本README.en.md
> - 文档内容：运行环境(软硬件)、数据集和模型文件路径、进一步处理/转换方式(如有)、运行信息参考


# Tensorflow2-NV-Resnet50 

## 1.运行环境
- 硬件: 
- 软件:
    - Tensorflow 2.6.0
    - Cuda compilation tools, release 11.4, V11.4.120
    - 具体信息参考路径: training/nvidia/docker_image/tensorflow2/Dockerfile

## 2.数据集准备

### TFRecords
下载ImageNet2012，并转化为TFRecord格式, 参考https://github.com/kwotsin/create_tfrecords 做数据处理。

## 2. 模型Checkpoint下载
在x链接下载第80个checkpoint, 并在training/nvidia/resnet50-tensorflow2/config/config_A100x1x8.py 中model_ckpt_dir给定路径, 即可从第80个epoch开始继续训练。

如果想要从头开始训练，将config_A100x1x8.py 中enable_backup_and_restore设为False 即可。

## 3. 运行信息参考

-  *//暂时不考虑资源占用情况等指标，后续monitor优化后再添加*

| 训练资源 | 配置文件 | 运行时长（大约） | 精度 | Steps数 | 性能（samples/秒） |
| -------- | -------- | ---------------- | ---- | ------- | ------- | 
| 单机8卡   |config_A100x1x8py | 1290    | 0.767  |    -    | 9928 |


## 4. 补充信息

```python
# 在tf中多机和单机采用不同的分布式方式, 在配置文件中提供两种配置参考案例:
# 多机
# runtime = dict(
#   distribution_strategy = 'multi_worker_mirrored',
#   run_eagerly = None,
#   tpu = None,
#   batchnorm_spatial_persistent = True)

# 单机
runtime = dict(distribution_strategy='mirrored',
               run_eagerly=None,
               tpu=None,
               batchnorm_spatial_persistent=True)

```