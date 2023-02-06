# 昆仑 Bert-Pytorch

- 语言：中文
- 厂商名称：昆仑芯
- 推荐环境配置
  - 硬件
    - 机器型号：昆仑芯 R480
    - 芯片/加速卡型号: 昆仑芯 R480
  - 软件
    - OS类型：Linux
    - OS kernel版本: Linux 5.4.0-26-generic x86_64
    - Docker 版本：例如 20.10.9
    - 加速卡监控命令：xpu_smi
- 厂商对于模型的适配
  - 算子扩展
  - 算子优化：基于MILR的算子融合、图优化
  - 数值精度：float32
  - 支持的分布式训练方式: 单卡、单机多卡

| 分布式训练方式 | 启动命令【含参数】 |
| :-----| :---- | 
| 单机单卡 | python -m torch.distributed.launch --nproc_per_node 1 --master_port=12355 run_pretraining.py --do_train --data_dir=dataset --extern_config_dir=/home/FlagPerf/training/kunlun/bert-pytorch/config --extern_config_file=config_R200x1x1.py --enable_extern_config |
| 单机多卡 | python -m torch.distributed.launch --nproc_per_node 2 --master_port=12355 run_pretraining.py --do_train --data_dir=dataset --extern_config_dir=/home/FlagPerf/training/kunlun/bert-pytorch/config --extern_config_file=config_R200x1x2.py --enable_extern_config |

  - 训练加速方式
  - 训练收敛情况
    - target_acc: 0.720
    - converged_acc: 0.739
    - epoch: 2
    - loss: 1.174
    - seed: 9031
    - Intial LR: 1.4e-4
    - batch-size: 12


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
