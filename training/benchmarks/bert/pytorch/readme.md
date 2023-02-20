# 模型信息与数据集、模型Checkpoint下载


● 模型Checkpoint下载地址：
`https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT`
   
```
文件列表：
tf1_ckpt
vocab.txt
bert_config.json
```

● 测试数据集下载地址：`https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v`

```
文件列表：
results_text.tar.gz
bert_reference_results_text_md5.txt
```

注：详情参考https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch
# 数据集和模型Checkpoint文件处理


● 模型格式转换：

```
git clone https://github.com/mlcommons/training_results_v1.0.git
cd training_results_v1.0/NVIDIA/benchmarks/bert/implementations/pytorch/
docker build --pull -t mlperf-nvidia:language_model .
```

启动容器，将checkpoint保存路径挂载为/cks

```
python convert_tf_checkpoint.py --tf_checkpoint /cks/model.ckpt-28252.index --bert_config_path /cks/bert_config.json --output_checkpoint model.ckpt-28252.pt
```


● 数据集格式转换：

```
cd /data && tar xf results_text.tar.gz
cd results4
md5sum --check ../bert_reference_results_text_md5.txt
cd ..
cp training_results_v1.0/NVIDIA/benchmarks/bert/implementations/pytorch/input_preprocessing/* ./
```

再次启动容器，将/data保存路径挂载为/data

```
cd /data
./parallel_create_hdf5.sh
mkdir -p 2048_shards_uncompressed
python3 ./chop_hdf5_files.py
mkdir eval_set_uncompressed

python3 create_pretraining_data.py \
  --input_file=results4/eval.txt \
  --output_file=eval_all \
  --vocab_file=vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_hdf5_file=eval_all.hdf5 \
  --output_hdf5_file=eval_set_uncompressed/part_eval_10k.hdf5 \
  --num_examples_to_pick=10000
```

# Nvidia GPU配置与运行信息参考
## 环境配置
### 硬件环境
  - NVIDIA A100 SXM 40GB
### 软件环境
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-113-generic
  - 加速卡驱动版本: 
    - Driver Version: 470.129.06
    - CUDA Version: 11.4
  - Docker 版本: 20.10.16
  - 训练框架版本：pytorch-1.8.0
  - 依赖软件版本：无
### 运行情况
| 训练资源 | 配置文件 | 运行时长(s) | final_loss | final_mlm_accuracy | Steps数 | 性能 (seq/s) |
| :-----| :---- | :---- | :---- | :---- | :---- | :---- |
| 单机单卡 | config_A100x1x1.py | 4616 | 1.1465 | 0.7442 | 25000 | 65.89 |
| 单机多卡 | config_A100x1x2.py | 4822 | 0.7836 | 0.8130 | 25000 | 125.57 |

