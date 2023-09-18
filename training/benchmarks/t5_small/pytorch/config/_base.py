# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "t5_small"
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
train_batch_size: int = 32
eval_batch_size: int = 32

max_epoch: int = 3
target_rouge1: float = 40.5

do_train = True
distributed: bool = True

# =========================================================
# utils
# =========================================================
seed: int = 0
dist_backend: str = 'nccl'
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 500
print_freq: int = 500
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1
