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
|                |—— throughput.py #测量吞吐量
|—— TTFT #用于观测首字延迟
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
|                |—— ttft.py #测量首字延迟
|—— tasks #配置任务信息
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
    5. nproc_per_node:采用的gpu数量
    6. vendor: 厂商名称
    7. config_path: 配置文件存放路径
2. vendor/engine/task.yaml
    1. GPU_NAME: GPU名称
    2. GPU_FPxx: 在xx精度下GPU的理论峰值FLOPs(单位为TFLOPs)
    3. task_nums: 任务数量

#### 运行方式
厂商修改完task.yaml和host.yaml文件后，调用python main.py 即可进行评测，评测结果会显示在控制台以及/log/engine/之中的log文件

