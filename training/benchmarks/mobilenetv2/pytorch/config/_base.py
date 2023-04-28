from typing import ClassVar
#from train.event.base import BaseTrainingEventInterface

# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
name: str = "MobileNetV2"

do_train = True
fp16 = True
# =========================================================
# data
# =========================================================
data_dir: str = None
train_data: str = "train"
eval_data: str = "val"
output_dir: str = ""
init_checkpoint: str = ""

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8
dist_backend: str = 'nccl'

lr: float = 0.045
lr_step_size: int = 1 
lr_gamma: float = 0.98

weight_decay: float = 0.00004
gradient_accumulation_steps: int = 1
momentum: float = 0.9

max_steps: int = 5005 * 300 # 300 epoch
seed: int = 41

# Stop training after reaching this accuracy
target_acc1: float = 70.973

# Sample to begin performing eval.
eval_iter_start_samples: int = 100

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_interval_samples: int = 5005 * 256 * 1 # 1 epoch

# Total number of training samples to run.
max_samples_termination: float = 5005 * 256 * 300 # 300 epoch

# number workers for dataloader
num_workers: int = 16

# local_rank for distributed training on gpus
local_rank: int = 0
# Whether to read local rank from ENVVAR
use_env: bool = True

# Number of epochs to plan seeds for. Same set across all workers.
num_epochs_to_generate_seeds_for: int = 2

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 10

# Whether to resume training from checkpoint.
# If set, precedes init_checkpoint/init_tf_checkpoint
resume_from_checkpoint: bool = False

# A object to provide some core components in training
#training_event: ClassVar[BaseTrainingEventInterface] = None

#training_event_instance: BaseTrainingEventInterface = None

# device
device: str = None
n_device: int = 1
