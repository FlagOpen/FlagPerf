from config_common import *


dist_backend = "xccl"

use_env = True
target_embedding_average = 0.92


gradient_accumulation_steps = 1

train_batch_size = 32
eval_batch_size = train_batch_size
max_steps = 10000

warmup = 0.2
learning_rate = 0.000125

beta_1: float = 0.9
beta_2: float = 0.99

opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999

eps: float = 1e-08

seed = 23333

skip_evaluator = False
debug_precision = False
debug_cpu = False
