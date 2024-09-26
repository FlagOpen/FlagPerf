### TTFT测量说明

#### 代码目录说明
```
├── README.md
|—— TTFT #用于观测首字延迟
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
                |—— ttft.py

```
#### 运行结果
运行过程中的输出会记录在/log/engine/ttft.log文件中。由于数据集不对齐且hugging face框架不支持continuous batch，因此在推理过程中需要padding，无法与vllm做到结果对齐，故不对huggingface框架下进行多并发的测量
#### 英伟达结果公开
|框架名称 |GPU名称 |并发数 |TTFT(s) |
|  :--- | :---: | :---: | ---: |
|vllm |A100-40G-SXM |256 |23.24|
|vllm |A100-40G-SXM |1   |0.09|
|huggingface |A100-40G-SXM |1 |0.75|


