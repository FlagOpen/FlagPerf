vendor:str = "nvidia"
data_dir: str = "/mnt/data/maskrcnn/train/"
train_batch_size = 16
eval_batch_size = 16


dist_backend = "nccl"
weight_decay = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-08
gradient_accumulation_steps = 1
warmup = 0.1
lr = 0.01
log_freq = 1
seed = 10483
max_samples_termination = 5553080
training_event = None
