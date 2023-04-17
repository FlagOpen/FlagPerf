mutable_params = [
    'data_dir', 'init_checkpoint', 'batch_size',
    'lr', 'gradient_accumulation_steps', "vendor"
]

mutable_params += ["local_rank", "do_train", "log_freq"]
