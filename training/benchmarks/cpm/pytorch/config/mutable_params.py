mutable_params = [
    "fp16", "dist_backend", "gradient_accumulation_steps", "train_batch_size",
    "eval_batch_size", "max_steps", "max_samples_termination", "warmup",
    "warmup_steps", "lr_decay_iters", "learning_rate", "weight_decay_rate",
    "beta_1", "beta_2", "eps", "seed", "target_embedding_average",
    "attention_dropout", "hidden_dropout", "loss_scale", "dynamic_loss_scale",
    "hysteresis", "loss_scale_window", "min_scale",
    "use_gradient_as_bucket_view", "num_workers", "vendor"
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq"]
