### 1. 推理数据集
> Download website：https://image-net.org/

We use ImageNet2012 Validation Images:
| Dataset                       | FileName               | Size  | Checksum                              |
| ----------------------------- | ---------------------- | ----- | ------------------------------------- |
| Validation images (all tasks) | ILSVRC2012_img_val.tar | 6.3GB | MD5: 29b22e2961454d5413ddabcf34fc5622 |
Dataset format conversion：
https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar are in the same directory with extract_ILSVRC.sh.
```bash
sh extract_ILSVRC.sh
```

preview directory structures of decompressed dataset.

```bash
tree -d -L 1
```

```
.
├── train
└── val
```
dataset samples size

```bash
find ./val -name "*JPEG" | wc -l
50000
```

### 2. 模型与权重

* 模型实现
  * pytorch：torchvision.models.resnet50
* 权重下载
  * pytorch：https://download.pytorch.org/models/resnet50-0676ba61.pth

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

- 推理工具包
   
   - XTCL 2.1

#### 2.3 天数智芯 MR-V100

- ##### 硬件环境
    - 机器、加速卡型号: MR-V100
    
- ##### 软件环境
   - OS版本：Ubuntu 18.04
   - OS kernel版本: 5.15.0-78-generic
   - 加速卡驱动版本：3.2.0
   - Docker 版本：24.0.4
   - 训练框架版本：torch-1.13.1+corex.3.2.0
   - 依赖软件版本：
     - cuda: 10.2
   
- 推理工具包

   - IXRT: ixrt-0.4.0+corex.3.2.0

#### 2.5 腾讯紫霄 C100

- ##### 硬件环境
    - 机器、加速卡型号: C100
    
- ##### 软件环境
   - OS版本：Ubuntu 20.04
   - OS kernel版本: 5.15.0-78-generic
   - 加速卡驱动版本：2.4.12
   - Docker 版本：24.0.4
   - 依赖软件版本：
     - pytorch: 1.13.0+cpu
     - onnx: 1.14.0
   
- 推理工具包

   - zxrt 2.4.12

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
| **计算卡使用率** | **\*MFU** | model flops utilization                             |
| 推理结果           | acc(推理/验证)   | 单位为top1分类准确率(acc1)                   |

* 指标值

| 推理工具  | precision | bs   | e2e_time | p_val_whole | p_val_core | p_infer_whole | \*p_infer_core | \*MFU     | acc         | mem        |
| ----------- | --------- | ---- | ---- | -------- | ----------- | ---------- | ------------- | ------------ | ----------- | ----------- |
| tensorrt | fp16      | 256  |613.4 | 1358.9   | 4469.4 | 1391.4   | 12698.7 | 16.8% | 76.2/76.2 | 19.7/40.0 |
| tensorrt | fp32   | 256  | 474.4    | 1487.3      | 2653.2     | 1560.3        | 6091.6  | 16.1% | 76.2/76.2 | 28.86/40.0 |
| torchtrt | fp16     | 256  | 716.4 | 1370.4 | 4282.6 | 1320.0 | 4723.0 | 6.3% | 76.2/76.2 | 9.42/40.0 |
| ixrt     | fp16  (W16A32)   | 256  | 261.467 | /      | /      | 1389.332  | 2721.402 | 11.7% | 76.2/76.2 | 8.02/32.0 |
| kunlunxin_xtcl | fp32   | 128  | 311.215    | /      | /     |  837.507    | 1234.727  | / | 76.2/76.2 | / |
| zixiao | fp16   | 32*6  | 261.103    | /      | /     |  193.151    | 6342.191  | / | 76.2/76.2 | / |

