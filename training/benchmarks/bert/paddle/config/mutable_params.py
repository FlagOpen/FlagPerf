mutable_params = [
    "train_batch_size", "eval_batch_size", "learning_rate", "weight_decay_rate", "opt_lamb_beta_1",
    "opt_lamb_beta_2", "max_steps", "max_samples_termination", "warmup_proportion", "warmup_steps",
    "start_warmup_step", "dist_backend", "seed", "gradient_accumulation_steps", "fp16",
    "loss_scale", "exchange_padding", "enable_fuse_dropout", "disable_fuse_mask", "fused_gelu_bias",
    "fused_dropout_add", "dense_seq_output"
    #"cache_eval_data"
]

mutable_params += [
    "use_cuda",
    "local_rank",
    "do_train",
    "data_dir",
    "log_freq"
]
