tokenizer_path = "tokenizer"
localbs = 3
train_steps = 100
theoryflops = 989000000000000.0
megatron_path = "/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 8
pipeline_parallel = 2