mutable_params = [
    'train_data', 'eval_data', 'init_checkpoint', 'train_batch_size',
    'eval_batch_size', 'dist_backend', 'vendor', 'local_rank', 'do_train',
    'data_dir', 'log_freq', 'output_dir', 'resume'
]
mutable_params += [
    'lr', 'lr_scheduler', 'lr_warmup_epochs', 'lr_warmup_method',
    'auto_augment', 'random_erase', 'label_smoothing', 'mixup_alpha',
    'cutmix_alpha', 'weight_decay', 'norm_weight_decay', 'ra_sampler',
    'ra_reps', 'epochs', 'num_workers', 'train_crop_size', 'val_crop_size',
    'val_resize_size', 'train_batch_size', 'eval_batch_size'
]
