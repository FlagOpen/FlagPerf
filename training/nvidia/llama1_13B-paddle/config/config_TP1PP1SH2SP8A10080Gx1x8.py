# model info
model_name_or_path: str = "facebook/llama-13b" 
tokenizer_name_or_path: str = "facebook/llama-13b"
continue_training = 0
split = "998,1,1"
max_seq_length = 2048

# training info
dataloader_num_workers = 1
max_steps = 100
save_steps = 10000
eval_steps = 10000
learning_rate = 3e-4
min_learning_rate = 3e-5
warmup_steps = 2000
weight_decay = 0.1
lr_scheduler_type = "cosine"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-06
max_grad_norm = 1.0
target_loss = 1.0
target_ppl = 0.6
logging_steps = 1
log_freq = 1
seed = 42

# for parallel
per_device_train_batch_size = 2
per_device_eval_batch_size = 1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
sharding_parallel_degree = 8
gradient_accumulation_steps = 128
use_flash_attention = 1
fuse_attention_qkv = 0
use_fused_rms_norm = 1
fp16 = True
fp16_opt_level = "O2"
scale_loss = 1024
sharding = "stage2"
recompute = True
recompute_granularity = "full"
