vendor = "kunlunxin"
dist_backend = "xccl"

lr = 0.5
lr_scheduler = "cosineannealinglr"
lr_warmup_epochs = 5
lr_warmup_method = "linear"
auto_augment = "ta_wide"
random_erase = 0.1
label_smoothing = 0.1
mixup_alpha = 0.2
cutmix_alpha = 1.0
weight_decay = 0.00002
norm_weight_decay = 0.0
model_ema = True
ra_sampler = True
ra_reps = 4
epochs = 600

# efficientnet_v2_s
TRAIN_SIZE = 300
train_crop_size = TRAIN_SIZE
EVAL_SIZE = 384
val_crop_size = EVAL_SIZE
val_resize_size = EVAL_SIZE