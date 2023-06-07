mutable_params = [
    'train_data', 'eval_data', 'init_checkpoint', 'train_batch_size',
    'eval_batch_size', 'dist_backend', 'lr', 'weight_decay',
    'gradient_accumulation_steps', 'max_samples_termination', "vendor",
    'cudnn_benchmark',
    'cudnn_deterministic'
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq"]
