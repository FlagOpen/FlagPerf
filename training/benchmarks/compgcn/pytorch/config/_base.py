# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = "nvidia"
data_dir: str = None
name: str = "compgcn"

# Optional paramters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "valid"
test_data: str = "test"

# =========================================================
# loss scale
# =========================================================
# =========================================================
# train && evaluate
# =========================================================
# target acc
target_MRR: float = 0.463
target_Hit1: float = 0.430

max_epochs: int = 500

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
