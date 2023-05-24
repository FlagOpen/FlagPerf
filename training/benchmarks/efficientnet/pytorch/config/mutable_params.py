mutable_params = [
    'train_data', 'eval_data', 'init_checkpoint', 'train_batch_size',
    'eval_batch_size', 'dist_backend', 'lr', 'weight_decay', "vendor"
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq", "output_dir"]
