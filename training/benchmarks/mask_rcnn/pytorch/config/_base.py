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
output_dir: str = "output"
init_checkpoint: str = ""

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8
dist_backend: str = 'nccl'

lr: float = 0.08
weight_decay: float = 1e-4
gradient_accumulation_steps: int = 1
momentum: float = 0.9
# steps for LR decay
lr_steps: list = [16, 22]
# decrease lr by a factor of lr-gamma
lr_gamma: float = 0.1

max_steps: int = 900000
seed: int = 41

# Stop training after reaching this accuracy
target_mAP: float = 0.58

# Sample to begin performing eval.
eval_iter_start_samples: int = 100

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_interval_samples: int = 100 * 256 * 1

# Total number of training samples to run.
max_samples_termination: float = 1388270 * 4 * 30

# number workers for dataloader
num_workers: int = 4

# local_rank for distributed training on gpus
local_rank: int = -1
# Whether to read local rank from ENVVAR
use_env: bool = True

# Number of epochs to plan seeds for. Same set across all workers.
num_epochs_to_generate_seeds_for: int = 2

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

# output path
output_dir: str = "output"

# Use sync batch norm
sync_bn: bool = False

gpu: int = None

# distributed training
distributed: bool = False

load_pretained: bool = True

# number of classes, background class NOT included
num_classes: int = 90

# pretrained path
pretrained_path: str = "checkpoint/resnet50.pth"
# coco weights pretrained_path
coco_weights_pretrained_path: str = "checkpoint/maskrcnn_resnet50_fpn_coco.pth"

# resume configs
# resume start checkpoint
resume: str = None
# start epoch
start_epoch: int = 0
