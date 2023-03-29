## 模型信息
### 模型介绍

BERT stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

Please refer to this paper for a detailed description of BERT:  
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 


###  模型代码来源
[Bert MLPerf](https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations)
 

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

### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | N/A |✅  |N/A|



