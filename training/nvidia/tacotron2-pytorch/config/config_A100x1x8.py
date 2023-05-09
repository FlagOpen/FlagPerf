from config_common import *

fp16 = True
dist_backend = "nccl"

gradient_accumulation_steps = 1

train_batch_size = 128
eval_batch_size = train_batch_size

max_steps = 4000
max_samples_termination = 4391260

warmup = 0.2
learning_rate = 0.0005

seed = 23333
