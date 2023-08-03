### 1. 推理数据集

● 下载地址：`https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v`

```
文件列表：
results_text.tar.gz
bert_reference_results_text_md5.txt
```

* 解压后将eval.txt放置在<data_dir>目录下

### 2. 模型与权重

* 模型实现
  * pytorch：transformers.BertForMaskedLM
* 权重下载
  * pytorch：BertForMaskedLM.from_pretrained("bert-large/base-uncased")
* 权重选择
  * 使用save_pretrained将加载的bert-large或bert-base权重保存到<data_dir>/<weight_dir>路径下

### 3. 软硬件配置与运行信息参考

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

### 4. 运行情况（BERT-Large）

* 指标列表

| 指标名称           | 指标值索引        | 特殊说明                                                    |
| ------------------ | ----------------- | ----------------------------------------------------------- |
| 数据精度           | precision         | 可选fp32/fp16                                               |
| 批尺寸             | bs                | 此外，对于bert-large seq_length==128                        |
| 批输入大小         | byte_per_batch    | 每一个batch包含的字节数                                     |
| 硬件存储使用       | mem               | 通常称为“显存”,单位为GiB                                    |
| 端到端时间         | e2e_time          | 总时间+Perf初始化等时间                                     |
| 验证总吞吐量       | p_val_whole       | 实际验证序列数除以总验证时间                                |
| 验证计算吞吐量     | *p_val_core       | 不包含IO部分耗时                                            |
| 推理总吞吐量       | p_infer_whole     | 实际推理序列数除以总推理时间                                |
| **推理计算吞吐量** | ***p_infer_core** | 不包含IO部分耗时。此外，此值*seq_length即为token per second |
| 推理单样本耗时     | infer_time        | 1/p_infer_core，单位为毫秒（ms）或微秒（μs）                |
| 推理结果           | acc(推理/验证)    | 单位为top1MaskedLM准确率(acc1)                              |

* 指标值

| 推理工具 | precision | bs   | byte_per_batch | e2e_time | p_val_whole | \*p_val_core | p_infer_whole | \*p_infer_core | acc         | mem       |
| -------- | --------- | ---- | -------------- | -------- | ----------- | ------------ | ------------- | -------------- | ----------- | --------- |
| tensorrt | fp16      | 32   | 32768          | 1283.9   | 257.3       | 260.4        | 408.3         | 418.1          | 0.600/0.638 | 17.4/40.0 |

