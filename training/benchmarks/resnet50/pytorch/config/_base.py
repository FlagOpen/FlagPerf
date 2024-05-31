# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "resnet50"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "val"

# =========================================================
# loss scale
# =========================================================
lr: float = 0.1
weight_decay: float = 1e-4
momentum: float = 0.9
lr_steps: list = 30
lr_gamma: float = 0.1

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 32
eval_batch_size: int = 32

target_acc1: float = 75.5
max_epoch: int = 100

do_train = True
fp16 = False
amp: bool = False
distributed: bool = True

# =========================================================
# utils
# =========================================================
seed: int = 0
dist_backend: str = 'nccl'
num_workers: int = 32
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
