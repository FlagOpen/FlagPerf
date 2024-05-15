tokenizer_path = "tokenizer"
localbs = 1
train_steps = 100
theoryflops = 312000000000000.0
megatron_path = "/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 4
pipeline_parallel = 1