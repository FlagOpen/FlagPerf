from config_common import *

fp16 = True
dist_backend = "nccl"

gradient_accumulation_steps = 1

train_batch_size = 128
eval_batch_size = train_batch_size

warmup = 0.2
learning_rate = 1e-3

seed = 23333
