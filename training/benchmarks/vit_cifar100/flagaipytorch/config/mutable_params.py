mutable_params = [
    'train_data', 'eval_data', 'batch_size', 'lr', 'weight_decay', 
    'gradient_accumulation_steps', 'warmup', 'seed', 'log_freq', 'fp16', 'max_epochs',
    'eval_interval', 'save_interval',
]

mutable_params += ["local_rank", "not_call_launch", "data_dir", ]
