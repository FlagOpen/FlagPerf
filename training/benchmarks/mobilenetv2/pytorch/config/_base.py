# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "mobilenetv2"
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
lr: float = 0.045
weight_decay: float = 0.00004
momentum: float = 0.9
lr_steps: list = 1
lr_gamma: float = 0.98

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8

# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L193
target_acc1: float = 68.6
# https://github.com/pytorch/vision/tree/main/references/classification
max_epoch: int = 300


do_train = True
fp16 = False
amp: bool = False
distributed: bool = True

# =========================================================
# utils
# =========================================================
seed: int = 41
dist_backend: str = 'nccl'
num_workers: int = 16
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 100
print_freq: int = 100
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1
