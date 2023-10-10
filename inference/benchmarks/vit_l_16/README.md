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
  * pytorch：transformers.ViTForImageClassification（hugging face）
* 权重下载
  * pytorch：from_pretrained("google/vit-large-patch16-224")（hugging face）

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
| tensorrt | fp16    | 64   |1009.7 | 777.8 | 796.7 | 825.8 | 1329.2 | 26.2% | 79.0/79.3 | 35.0/40.0 |
| tensorrt | fp32   | 32 | 1275.9 | 482.4  | 491.1 | 555.5    | 590.5 | 23.3% | 79.3/79.3 | 35.0/40.0 |
| kunlunxin_xtcl | W32A16   | 32 | 2118.307 | / | / | 130.006    | 144.914 | 27.9% | 79.3/79.3 | / |

