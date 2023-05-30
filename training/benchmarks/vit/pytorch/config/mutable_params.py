mutable_params = [
    'train_data', 'eval_data', 'init_checkpoint', 'train_batch_size',
    'eval_batch_size', 'dist_backend', 'vendor', 'local_rank', 'do_train',
    'data_dir', 'log_freq', 'output_dir', 'resume'
]
mutable_params += [
    'epochs', 'opt', 'lr', 'wd', 'lr_scheduler', 'lr_warmup_method',
    'lr_warmup_epochs', 'lr_warmup_decay', 'amp', 'label_smoothing',
    'mixup_alpha', 'auto_augment', 'clip_grad_norm', 'ra_sampler', 
    'cutmix_alpha'
]
