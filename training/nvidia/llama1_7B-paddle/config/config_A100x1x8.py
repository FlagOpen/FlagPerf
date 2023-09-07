# stage2
# model_name_or_path: str = "facebook/llama-7b" 
# tokenizer_name_or_path: str = "facebook/llama-7b"
# split = "949,50,1"
# max_seq_length = 2048
# per_device_train_batch_size = 2
# per_device_eval_batch_size = 2
# use_flash_attention = True
# use_fused_rms_norm = False 
# fp16 = True
# fp16_opt_level = "O2"
# scale_loss = 1024
# learning_rate = 0.0001
# min_learning_rate = 0.00001
# max_steps = 15
# save_steps = 5000
# weight_decay = 0.01
# warmup_ratio = 0.01
# max_grad_norm = 1.0
# logging_steps = 1
# log_freq = logging_steps
# dataloader_num_workers = 1
# sharding = "stage2"
# eval_steps = 15
# disable_tqdm = True
# continue_training = False
# recompute = False

# tp2 pp4
model_name_or_path: str = "facebook/llama-7b" 
tokenizer_name_or_path: str = "facebook/llama-7b"
split = "949,50,1"
max_seq_length = 2048
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
use_flash_attention = True
use_fused_rms_norm = False 
fp16 = True
fp16_opt_level = "O2"
tensor_parallel_degree 2
pipeline_parallel_degree 4
scale_loss = 1024
learning_rate = 0.0001
min_learning_rate = 0.00001
max_steps = 20
save_steps = 5000
weight_decay = 0.01
warmup_ratio = 0.01
max_grad_norm = 1.0
logging_steps = 1
log_freq = logging_steps
dataloader_num_workers = 1
do_eval = False
disable_tqdm = True
continue_training = False
recompute = False

# tp=2 stage1=4
model_name_or_path: str = "facebook/llama-7b" 
tokenizer_name_or_path: str = "facebook/llama-7b"
split = "949,50,1"
max_seq_length = 2048
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
use_flash_attention = True
use_fused_rms_norm = False 
fp16 = True
fp16_opt_level = "O2"
tensor_parallel_degree 2
sharding = "stage1"
scale_loss = 1024
learning_rate = 0.0001
min_learning_rate = 0.00001
max_steps = 20
save_steps = 5000
weight_decay = 0.01
warmup_ratio = 0.01
max_grad_norm = 1.0
logging_steps = 1
log_freq = logging_steps
dataloader_num_workers = 1
do_eval = False
disable_tqdm = True
continue_training = False
recompute = False