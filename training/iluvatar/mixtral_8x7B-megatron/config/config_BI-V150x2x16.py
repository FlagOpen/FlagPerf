mixtral_iluvatar_path = "/data1/user_homes/gengyang/code/FlagPerf/data_dir"
tokenizer_path = mixtral_iluvatar_path + "/flagscale-iluvatar-mixtral/data_dir/Qwen1___5-7B-Chat-GPTQ-Int8"
localbs = 1  #micro-batch-size
train_steps = 100  #
theoryflops = 192000000000000.0
megatron_path = mixtral_iluvatar_path + "/flagscale-iluvatar-mixtral"#"/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 2  #四机为4，暂时设置为2
pipeline_parallel = 2 
