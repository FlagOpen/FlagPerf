# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = "nvidia"
data_dir: str = None
name: str = "bigtransfer"

# torch.backends.cudnn.benchmark
cudnn_benchmark: bool = False
# torch.backends.cudnn.deterministic
cudnn_deterministic: bool = True

# Optional paramters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "val"
transfered_weight: str = "backbone_weights/bigtransfer/"

# =========================================================
# model
# =========================================================
model_shard: int = 2

# =========================================================
# loss scale
# =========================================================
lr: float = 0.003
momentum: float = 0.9
warmup_steps: int = 500
lr_steps: list = [3000, 6000, 9000, 10000]
lr_gamma: float = 0.1

# =========================================================
# train && evaluate
# =========================================================
batch_size: int = 16
train_batch_size: int = batch_size
eval_batch_size: int = batch_size

target_mAP: float = 0.83
max_steps: int = 40000
# step per epoch=1281167/global batchsize
# 40000 steps with bs==24 1*8 are 4 epoch
# indeed, transfer model should only fine-tune 1 epoch

fp16 = False
distributed: bool = True
# =========================================================
# utils
# =========================================================
seed: int = 0


dist_backend: str = 'nccl'
num_workers: int = 8
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 20
print_freq: int = 20
n_device: int = 1
amp: bool = False
sync_bn: bool = False
gradient_accumulation_steps: int = 1
