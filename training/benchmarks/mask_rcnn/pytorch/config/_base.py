# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
name: str = "Mask_RCNN"

# train/eval flag
do_train = True

# fp16 training flag
fp16 = False
# =========================================================
# data
# =========================================================
data_dir: str = None
train_data: str = "train"
eval_data: str = "val"
init_checkpoint: str = ""

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8
dist_backend: str = 'nccl'

gradient_accumulation_steps: int = 1

lr: float = 0.08
weight_decay: float = 1e-4
momentum: float = 0.9
# steps for LR decay
lr_steps: list = [16, 22]
# decrease lr by a factor of lr-gamma
lr_gamma: float = 0.1

seed: int = 1234

# Stop training after reaching this accuracy

target_map_bbox: float = 0.3558
target_map_segm: float = 0.3185

max_epochs: int = 26

# number workers for dataloader
num_workers: int = 4

# local_rank for distributed training on gpus
local_rank: int = -1
# Whether to read local rank from ENVVAR
use_env: bool = True

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 10

# print frequency
print_freq: int = 50

# device
device: str = None
# num of device
n_device: int = 1

# Automatic mixed precision
amp: bool = True
# aspect ratio group factor
aspect_ratio_group_factor: int = 3
# Use sync batch norm
sync_bn: bool = False

gpu: int = None

# distributed training
distributed: bool = False

load_pretained: bool = True

# number of classes, background class NOT included
num_classes: int = 90

# backbone pretrained path
pretrained_path: str = "checkpoint/resnet50.pth"
