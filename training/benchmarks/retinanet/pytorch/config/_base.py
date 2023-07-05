# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = "nvidia"
data_dir: str = None
name: str = "retinanet"

# Optional parameters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "val"

# use torch/hub/checkpoints/resnet50-0676ba61 as backbone weights
# no init weights for other parts of retinanet

# =========================================================
# Model
# =========================================================
aspect_ratio_group_factor: int = 3
num_classes: int = 90

# =========================================================
# loss scale
# =========================================================
lr: float = 0.08
weight_decay: float = 1e-4
momentum: float = 0.9
lr_steps: list = [16, 22]
lr_gamma: float = 0.1

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 16
eval_batch_size: int = 16

target_mAP: float = 0.35
max_epoch: int = 26

do_train = True
fp16 = False
distributed: bool = True
warmup = 0.1

# =========================================================
# utils
# =========================================================
seed: int = 0
dist_backend: str = 'nccl'
num_workers: int = 4
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 100
print_freq: int = 100
n_device: int = 1
amp: bool = False
sync_bn: bool = False
gradient_accumulation_steps: int = 1

cudnn_benchmark: bool = True
cudnn_deterministic: bool = False