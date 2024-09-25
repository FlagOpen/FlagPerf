### TTFT测量说明

#### 代码目录说明
```
├── README.md
|—— Throughput #用于观测吞吐量
|    ├── vendor #厂商配置
|          ├── engine #推理引擎
                |—— throughput.py

```
#### 运行结果
运行过程中的输出会记录在/log/engine/throughput.log文件中
#### 英伟达结果公开
|框架名称 |GPU名称 |并发数 |Tps |Throughput |MFU |
|  :--- | :---: | :---: | :---: | ---: | ---: |
|vllm |A100-40G-SXM |256 |5101.72|3848.26|28.84%
|huggingface |A100-40G-SXM |1 |650.87|21.58|3.68%


