# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "bert_hf"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# data
# =========================================================
seq_length = 512
mask_ratio = 0.10
weight_dir = "weights"
datafilename = "openwebtext_bert_100M.npy"
valdatafilename = "openwebtext_bert_10M.npy"
train_val_ratio = 0.1

# =========================================================
# loss scale
# =========================================================
lr: float = 0.00005

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8
gradient_accumulation_steps: int = 4

target_acc1: float = 0.655
max_epoch: int = 3

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
log_freq: int = 100
print_freq: int = 100
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1
