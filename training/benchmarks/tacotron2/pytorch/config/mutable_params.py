"""mutable_params defines parameters that can be replaced by vendor"""
mutable_params = [
    "amp",
    "dist_backend",
    "gradient_accumulation_steps",
    "train_batch_size",
    "eval_batch_size",
    "max_steps",
    "max_samples_termination",
    "learning_rate",
    "weight_decay",
    "seed",
    "loss_scale",
    "loss_scale_window",
    "min_scale",
    "num_workers",
    "distributed",
    "init_checkpoint",
    "vendor",
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq"]