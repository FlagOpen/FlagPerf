tokenizer_path = "llama3_8b_hf"
localbs = 1
train_steps = 300
theoryflops = 989000000000000.0
megatron_path = "/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 2
pipeline_parallel = 1
