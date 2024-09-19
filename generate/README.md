### 部署说明

#### 代码目录说明
```
├── README.md
├── main.py #用户调用接口
├── utils #用于分析的工具
|    ├── analyze.py #分析配置信息
|    ├── result_show.py #分析结果信息
|—— Throughput #用于观测吞吐量信息
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
|—— TTFT #用于观测首字延迟
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
|—— TASK #配置任务信息
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
|                 |—— GPUConfig.yaml #硬件信息配置
|—— host.yaml #路径信息配置

```
#### 数据集
1. 本次采用的是开源数据集XSum，该数据集侧重于模型对文本摘要的生成，偏重于模型推理。
2. 数据集下载地址 https://huggingface.co/datasets/knkarthick/xsum/tree/main，采用的是该仓库中全部数据作为测试集。https://github.com/EdinburghNLP/XSum/tree/master/ 源数据是以.summar形式存储，如果使用源数据集则需要按照前链接的数据集形式进行转换并以csv文件的形式存储
3. 数据集评测方式：采用ROUGE分数对推理结果进行评测，同原论文https://arxiv.org/abs/1808.08745 的测量方式保持一致。
#### 配置文件说明
1. host.yaml
    1. model_path: 模型存放路径
    2. data_path: 数据集存放路径
    3. log_path: 推理结果存放路径
    4. engine: 推理框架(支持扩展，现版本支持vllm以及huggingface框架下的推理)
    5. gpu_nums:采用的gpu数量
    6. vendor: 厂商名称
    7. config_path: 配置文件存放路径
2. vendor/engine/task.yaml
    1. GPU_NAME: GPU名称
    2. GPU_FPxx: 在xx精度下GPU的理论峰值FLOPs(单位为TFLOPs)
    3. task_nums: 任务数量

#### 运行方式
厂商修改完task.yaml和host.yaml文件后，调用python main.py 即可进行评测，评测结果会显示在控制台以及/log/engine/之中的log文件
#### 运行结果
运行结果会显示在屏幕中，如：
2024-09-19 05:56:48.920 | INFO     | __main__:<module>:37 - TTFT:0.7435413458806579
2024-09-19 05:56:48.920 | INFO     | __main__:<module>:38 - Throughput:21.241251614991388
2024-09-19 05:56:48.920 | INFO     | __main__:<module>:39 - Tps:640.5137188603395
2024-09-19 05:56:48.920 | INFO     | __main__:<module>:40 - Time:193.57586941402406
2024-09-19 05:56:48.920 | INFO     | __main__:<module>:41 - MFU:0.0036210430230075204
2024-09-19 05:56:48.921 | INFO     | __main__:<module>:42 - ROUGE1:0.10941584374196911
2024-09-19 05:56:48.921 | INFO     | __main__:<module>:43 - ROUGE2:0.002265168149848077
运行过程中的输出会记录在output.txt文件中
如：2024-09-19 06:22:31,864	INFO worker.py:1783 -- Started a local Ray instance.
INFO 09-19 06:22:41 llm_engine.py:72] Initializing an LLM engine with config: model='/raid/llama3_infer/llama3_70b_hf', tokenizer='/raid/llama3_infer/llama3_70b_hf', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=8, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, seed=0)
INFO 09-19 06:23:40 llm_engine.py:322] # GPU blocks: 23500, # CPU blocks: 6553
INFO 09-19 06:23:44 model_runner.py:632] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 09-19 06:23:44 model_runner.py:636] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
[36m(RayWorkerVllm pid=83185)[0m INFO 09-19 06:23:44 model_runner.py:632] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
[36m(RayWorkerVllm pid=83185)[0m INFO 09-19 06:23:44 model_runner.py:636] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 09-19 06:23:55 custom_all_reduce.py:199] Registering 5635 cuda graph addresses
[36m(RayWorkerVllm pid=83185)[0m INFO 09-19 06:23:55 custom_all_reduce.py:199] Registering 5635 cuda graph addresses
[36m(RayWorkerVllm pid=84278)[0m INFO 09-19 06:23:44 model_runner.py:632] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.[32m [repeated 6x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(RayWorkerVllm pid=84278)[0m INFO 09-19 06:23:44 model_runner.py:636] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.[32m [repeated 6x across cluster][0m
INFO 09-19 06:23:55 model_runner.py:698] Graph capturing finished in 11 secs.
[36m(RayWorkerVllm pid=83185)[0m INFO 09-19 06:23:55 model_runner.py:698] Graph capturing finished in 11 secs.

Processed prompts:   0%|          | 0/5000 [00:00<?, ?it/s]
Processed prompts:   0%|          | 1/5000 [00:00<11:13,  7.42it/s]
Processed prompts:   0%|          | 3/5000 [00:01<36:56,  2.25it/s]
Processed prompts:   0%|          | 4/5000 [00:01<27:28,  3.03it/s]
Processed prompts:   0%|          | 6/5000 [00:01<16:27,  5.06it/s]
Processed prompts:   0%|          | 7/5000 [00:01<17:00,  4.89it/s]
Processed prompts:   0%|          | 12/5000 [00:01<07:51, 10.57it/s]
