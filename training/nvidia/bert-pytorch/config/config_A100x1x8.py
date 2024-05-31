from config_Ampere_common import *

gradient_accumulation_steps = 4
start_warmup_step = 0
warmup_proportion = 0
warmup_steps = 0

distributed_lamb = False
learning_rate = 0.00035
weight_decay_rate = 0.01
opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999

eval_batch_size = train_batch_size
max_samples_termination = 4500000
cache_eval_data = True
max_steps = 30000

seed = 9031
