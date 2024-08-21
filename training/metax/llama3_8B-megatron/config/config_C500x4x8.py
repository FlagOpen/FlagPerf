tokenizer_path = "llama3_8b_hf"
localbs = 1
train_steps = 300
theoryflops = 280000000000000.0
megatron_path = "/workspace/Megatron-LM_metax" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 1
pipeline_parallel = 2
