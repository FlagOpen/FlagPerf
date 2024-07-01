tokenizer_path = "/data1/user_homes/lisen/codes/FlagPerf/training/benchmarks/llama3_8B/megatron-deepspeed/tokenizer_llama3.model"
localbs = 1
train_steps = 300
theoryflops = 192000000000000.0
megatron_path = "/workspace/megatron-deepspeed" # need to be aligned with DockerFile. In iluvatar, it's /workspace/ + Megatron-LM
tensor_parallel = 1
pipeline_parallel = 8
