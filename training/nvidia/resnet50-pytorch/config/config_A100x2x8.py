# from extern.training_even import ApexTrainingEvent
from config_common import *

ddp_type = "apex"
dist_backend = "nccl"

gradient_accumulation_steps = 1

train_batch_size = 256
eval_batch_size = train_batch_size
max_steps = 1000000

warmup = 0.2
learning_rate = 0.01

beta_1: float = 0.9
beta_2: float = 0.99
eps: float = 1e-08

seed = 23333
max_samples_termination = 43912600
training_event = None
