### Throughput测量说明

#### 代码目录说明
```
├── README.md
|—— Throughput #用于观测吞吐量
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
                |—— throughput.py

```
#### 运行结果
运行过程中的输出会记录在/log/engine/throughput.log文件中。由于数据集不对齐且hugging face框架不支持continuous batch，因此在推理过程中需要padding，无法与vllm做到结果对齐，故不对huggingface框架下进行多并发的测量。
#### 英伟达结果公开
|框架名称 |GPU名称 |并发数 |Tps |Throughput |MFU |
|  :--- | :---: | :---: | :---: | ---: | ---: |
|vllm |A100-40G-SXM |256 |5101.72|3848.26|28.84%|
|vllm |A100-40G-SXM |1 |1115.35|36.84|6.31%|
|huggingface |A100-40G-SXM |1 |650.87|21.58|3.68%|


