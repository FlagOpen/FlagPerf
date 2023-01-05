
### 模型Checkpoint下载

● 下载地址：
`https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT`
   

```
文件列表：
tf1_ckpt
vocab.txt
bert_config.json
```


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

### 测试数据集下载

● 下载地址：`https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v`

```
文件列表：
results_text.tar.gz
bert_reference_results_text_md5.txt
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

> 注：详情参考https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch

### Pytorch版本运行指南

● 运行脚本:

在该路径目录下

```
python run_pretraining.py 
--data_dir data_path
--extern_config_dir config_path
--extern_config_file config_file.py
```

example：
```
python run_pretraining.py 
--extern_config_dir /FlagPerf/training/kunlun/bert-pytorch/config
--extern_config_file config_R200x1x1.py
```


### 许可证

本项目基于Apache 2.0 license。
本项目部分代码基于MLCommons https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA 实现。
