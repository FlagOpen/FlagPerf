# case info
vendor: str = "nvidia"
data_dir: str = None
name: str = "detr"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "val"


# =========================================================
# Model
# =========================================================
lr = 1e-4
lr_backbone=1e-5
lr_drop = 200
weight_decay = 1e-4
clip_max_norm = 0.1
model_name = 'transformer'
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1
nheads = 8
num_queries = 100
pre_norm = False
masks = False


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
#Disables auxiliary decoding losses (loss at each layer)
aux_loss = True
# Class coefficient in the matching cost
set_cost_class = 1
# L1 box coefficient in the matching cost
set_cost_bbox = 5
# giou box coefficient in the matching cost
set_cost_giou = 2
mask_loss_coef = 1
dice_loss_coef = 1
bbox_loss_coef = 5
giou_loss_coef = 2
# Relative classification weight of the no-object class
eos_coef = 0.1


# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 2
eval_batch_size: int = 2

target_mAP: float = 0.35

start_epoch = 0
epochs = 10

do_train = True
fp16 = False
distributed: bool = True
warmup = 0.1


# =========================================================
# utils
# =========================================================
dataset_file = 'coco'
output_dir = ''
data_dir = ''
coco_path = ''
seed: int = 42
dist_backend: str = 'nccl'
num_workers: int = 1
device : str = None
resume =''

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


# -----------------------------------------------------------------------------
# distributed training parameters
# -----------------------------------------------------------------------------
local_rank: int = -1
use_env: bool = True
log_freq: int = 100
print_freq: int = 100
n_device: int = 1
amp: bool = False
sync_bn: bool = False
gradient_accumulation_steps: int = 1
