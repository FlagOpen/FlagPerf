# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "distilbert"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# loss scale
# =========================================================
lr: float = 5e-5
weight_decay = 0.0

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 4
eval_batch_size: int = 4

max_epoch: int = 3
target_acc: float = 40.5

do_train = True
distributed: bool = True

eval_iter_start_samples: int = 0
eval_interval_samples: int = 4

max_steps: int = 4000

# =========================================================
# utils
# =========================================================
seed: int = 0
dist_backend: str = 'nccl'
device: str = None

# =========================================================
# datasets
# =========================================================
dataloader_drop_last: bool = False
dataloader_num_workers: int = 8

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 1
print_freq: int = 1
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1
