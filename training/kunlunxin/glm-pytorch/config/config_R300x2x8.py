vendor = 'kunlunxin'
fp16 = False

train_batch_size = 4
eval_batch_size = 6

dist_backend = "xccl"

lr = 1e-5
weight_decay = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-08
gradient_accumulation_steps = 1
warmup = 0.1
lr_decay_ratio = 0.1
lr_decay_iters = 4338
log_freq = 1
seed = 4096
max_samples_termination = 5553080
training_event = None
