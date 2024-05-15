tokenizer_path = "/home/chenglongkai/datasets/mistral/tokenizer.model"
localbs = 1
train_steps = 2
theoryflops = 312000000000000.0
megatron_path = "/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 8
pipeline_parallel = 2