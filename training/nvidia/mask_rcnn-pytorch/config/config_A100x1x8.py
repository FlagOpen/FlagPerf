vendor = "nvidia"
dist_backend = "nccl"

train_batch_size = 16
eval_batch_size = 16

gradient_accumulation_steps = 1
warmup = 0.1
lr = 0.01
log_freq = 1
seed = 10483
