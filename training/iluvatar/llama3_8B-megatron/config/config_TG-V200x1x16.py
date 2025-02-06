tokenizer_path = "/home/zhiyuan/test/llama3-8b/tokenizer.model"
localbs = 1  #micro-batch-size
train_steps = 300  ##训练迭代次数
theoryflops = 276000000000000.0 ##由于测试环境功率限制，设定较低主频，限制理论算力上限，与官方标称算力不一致
megatron_path = "/usr/local/lib/python3.10/dist-packages/megatron" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 1  
pipeline_parallel = 4 
