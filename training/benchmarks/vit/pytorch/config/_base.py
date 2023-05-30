# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
name: str = "EfficientNet"

do_train = True
fp16 = False
# =========================================================
# data
# =========================================================
data_dir: str = None
train_data: str = "train"
eval_data: str = "val"
output_dir: str = ""
init_checkpoint: str = ""
resume: str = ""

# =========================================================
# train && evaluate
# reference to https://github.com/pytorch/vision/tree/main/references/classification
# =========================================================
train_batch_size: int = 256
eval_batch_size: int = 256
dist_backend: str = 'nccl'

# number of total epochs to run
epochs: int = 90

# number workers for dataloader
num_workers: int = 16

# optimizer
opt: str = 'sgd'

# initial learning rate
lr: float = 0.1

# momentum
momentum: float = 0.9

# weight decay
weight_decay: float = 1e-4

# weight decay for Normalization layers (default: None, same value as weight_decay)
norm_weight_decay: float = None

# weight decay for bias parameters of all layers (default: None, same value as weight_decay)
bias_weight_decay: float = None

# weight decay for embedding parameters for vision transformer models (default: None, same value as weight_decay)
transformer_embedding_decay: float = None

# label smoothing
label_smoothing: float = 0.0

# mixup alpha
mixup_alpha: float = 0.0

# cutmix alpha
cutmix_alpha: float = 0.0

# the lr scheduler
lr_scheduler: str = "steplr"

# the number of epochs to warmup
lr_warmup_epochs: int = 0

# the warmup method
lr_warmup_method: str = "constant"

# the decay for lr
lr_warmup_decay: float = 0.01

# decrease lr every step-size epochs
lr_step_size: int = 30

# decrease lr by a factor of lr_gamma
lr_gamma: float = 0.1

# minimum lr of lr schedule
lr_min: float = 0.0

# Use sync batch norm
sync_bn: bool = False

# auto augment policy
auto_augment: str = None

# magnitude of auto augment policy
ra_magnitude: int = 9

# severity of augmix policy
augmix_severity: int = 3

# random erasing probability
random_erase: float = 0.0

# Use torch.cuda.amp for mixed precision training
amp: bool = False

# the interpolation method
interpolation: str = "bilinear"

# the resize size used for validation
val_resize_size: int = 256

# the central crop size used for validation
val_crop_size: int = 224

# the random crop size used for training
train_crop_size: int = 224

# the maximum gradient norm
clip_grad_norm: float = None

# whether to use Repeated Augmentation in trainin
ra_sampler: bool = False

# number of repetitions for Repeated Augmentation
ra_reps: int = 3

seed: int = 41

# Stop training after reaching this accuracy
target_acc1: float = 81.072

# Sample to begin performing eval.
eval_iter_start_samples: int = 100

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_interval_samples: int = 1281167  # 1 epoch

# Total number of training samples to run.
max_samples_termination: float = 1281167 * 600  # 600 epoch

# local_rank for distributed training on gpus
local_rank: int = 0
# Whether to read local rank from ENVVAR
use_env: bool = True

# Number of epochs to plan seeds for. Same set across all workers.
num_epochs_to_generate_seeds_for: int = 600

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 10

# Whether to resume training from checkpoint.
# If set, precedes init_checkpoint/init_tf_checkpoint
resume_from_checkpoint: bool = False

gradient_accumulation_steps = 1

# device
device: str = None
n_device: int = 1
