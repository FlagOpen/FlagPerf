# model info
model_name_or_path = "gpt3-6.7B-en"
tokenizer_name_or_path = "gpt3-6.7B-en"
continue_training = 0
split = "949,50,1"
max_seq_length = 2048

# training info
dataloader_num_workers = 8
gradient_accumulation_steps = 32
max_steps = 512
save_steps = 10000
eval_steps = 5000000
learning_rate = 1.2e-4
min_learning_rate = 1.2e-5
weight_steps = 188
decay_steps = 130*1024*1024
max_grad_norm = 1.0
target_loss = 1.0
target_ppl = 0.6
logging_steps = 1
log_freq = 1
seed = 42

# for parallel
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
sharding_parallel_degree = 8
use_flash_attention = 1
fuse_attention_qkv = 0
use_fused_rms_norm = 1
fp16 = True
fp16_opt_level = "O2"
sharding = "stage2"
recompute = False
recompute_granularity = "full"
