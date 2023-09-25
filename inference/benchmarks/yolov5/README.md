### 1. 推理数据集
> Download website：https://cocodataset.org/

We use COCO2017 Validation Images:
| Dataset                       | FileName               | Size  | Checksum                              |
| ----------------------------- | ---------------------- | ----- | ------------------------------------- |
| Validation images (all tasks) | coco2017 | 1GB | / |

preview directory structures of decompressed dataset.

```bash
tree -d -L 1
```

```
├── annotations
├── train2017
└── val2017
```
dataset samples size

```bash
find ./val -name "*JPEG" | wc -l
5000
```

### 2. 模型与权重

* 模型实现
  * pytorch：yolov5l-bs96.onnx
* 权重下载
  * pytorch：

### 2. 软硬件配置与运行信息参考

#### 2.1 Nvidia A100

- ##### 硬件环境
    - 机器、加速卡型号: NVIDIA_A100-SXM4-40GB
    - 多机网络类型、带宽: InfiniBand，200Gb/s
    
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.4.0-113-generic
   - 加速卡驱动版本：470.129.06
   - Docker 版本：20.10.16
   - 训练框架版本：pytorch-1.13.0a0+937e930
   - 依赖软件版本：
     - cuda: 11.8
   
- 推理工具包

   - TensorRT 8.5.1.7
   - torch_tensorrt 1.3.0

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
     - pycocotools: 2.0.7

- 推理工具包

   - XTCL 2.1

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
| 推理结果           | acc(推理/验证)   | 单位为top1分类准确率(acc1)                   |

* 指标值

| 推理工具  | precision | bs   | e2e_time | p_val_whole | \*p_val_core | p_infer_whole | \*p_infer_core |\*MFU| acc         | mem        |
| ----------- | --------- | ---- | -------- | ----------- | ---------- | ------------- | ------------ |  ------------ |----------- | ---------- |
| tensorrt | fp32   | 96  | 733.8    |    /   | /    | 53.8       | 361.4 |12.6%| 0.45 | 35.44/40.0 |
| tensorrt | fp16   | 96  | 1665.8    |    /   | /    | 58.6     | 859 |15.0%| 0.45 | 26.15/40.0 |
| kunlunxin_xtcl | fp32   | 96  | / |    /   | / | /   | / |18.9%| 0.451 | 26.42/32.0 |
