dist_backend = "xccl"

target_mlm_accuracy = 0.67
gradient_accumulation_steps = 7
max_steps = 50000
start_warmup_step = 0
warmup_proportion = 0
warmup_steps = 0

learning_rate = 4e-4
weight_decay_rate = 0.01
opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999
train_batch_size = 8
eval_batch_size = train_batch_size
max_samples_termination = 4500000
cache_eval_data = False

seed = 9031
