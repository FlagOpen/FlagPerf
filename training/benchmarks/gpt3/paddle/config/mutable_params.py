
mutable_params = [
    "split", "max_seq_length", "per_device_train_batch_size", "per_device_eval_batch_size",
    "use_flash_attention", "use_fused_rms_norm", "fp16", "fp16_opt_level", "gradient_accumulation_steps",
    "max_steps", "eval_steps", "learning_rate", "min_learning_rate", "weight_decay", "warmup_ratio",
    "seed"
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq"]