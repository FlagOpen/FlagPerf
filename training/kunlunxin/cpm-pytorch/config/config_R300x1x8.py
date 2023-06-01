from config_common import *

dist_backend = "xccl"

train_batch_size = 32
eval_batch_size = train_batch_size
max_steps = 4000
max_samples_termination = 4391260

warmup = 0.2
learning_rate = 0.0005

seed = 23333
