from config_common import *

fp16 = True
dist_backend = "nccl"
target_embedding_average = 0.92

gradient_accumulation_steps = 1

train_batch_size = 32
eval_batch_size = train_batch_size
max_steps = 4000
max_samples_termination = 4391260

warmup = 0.2
learning_rate = 0.0005

beta_1: float = 0.9
beta_2: float = 0.99
eps: float = 1e-08

seed = 23333
training_event = None