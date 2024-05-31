mutable_params = [
    'train_data', 'eval_data', 'init_checkpoint', 'train_batch_size',
    'eval_batch_size', 'dist_backend', 'lr', 'weight_decay', 'adam_beta1',
    'adam_beta2', 'adam_eps', 'gradient_accumulation_steps', 'warmup',
    'lr_decay_ratio', 'lr_decay_iters', 'max_samples_termination', "vendor"
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq"]
