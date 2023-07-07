vendor = "kunlunxin"
dist_backend = "xccl"

epochs = 300
opt = "adamw"
lr = 0.003
weight_decay = 0.3
lr_scheduler = "cosineannealinglr"
lr_warmup_method = "linear" 
lr_warmup_epochs = 30
lr_warmup_decay = 0.033 
amp = False
label_smoothing = 0.11
mixup_alpha = 0.2
auto_augment = "ra"
clip_grad_norm = 1
ra_sampler = True
cutmix_alpha = 1.0
