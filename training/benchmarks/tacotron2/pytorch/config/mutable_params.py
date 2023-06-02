"""mutable_params defines parameters that can be replaced by vendor"""
mutable_params = [
    "amp",
    "dist_backend",
    "train_batch_size",
    "eval_batch_size",
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

mutable_params += ["local_rank", "do_train", "data_dir"]