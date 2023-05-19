from typing import ClassVar
#from train.event.base import BaseTrainingEventInterface

# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
# name: str = "MobileNetV2"
name: str = "yolov5"

do_train = True
fp16 = True
# =========================================================
# data
# =========================================================
# data_dir: str = None
# train_data: str = "train"
# eval_data: str = "val"
output_dir: str = ""
init_checkpoint: str = ""

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8
dist_backend: str = 'nccl'

lr: float = 0.1
weight_decay: float = 1e-4
gradient_accumulation_steps: int = 1
momentum: float = 0.9

max_steps: int = 600000
seed: int = 41

# Stop training after reaching this accuracy
target_acc1: float = 50.0

# Sample to begin performing eval.
eval_iter_start_samples: int = 100

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_interval_samples: int = 100 * 256 * 1

# Total number of training samples to run.
max_samples_termination: float = 1388270 * 4 * 30

# number workers for dataloader
num_workers: int = 4

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




cfg: str = "/data/sen.li/workspace/code/FlagPerf/training/benchmarks/yolov5/pytorch/model/yolov5s.yaml"
# hpy = "path/to/hpy.yaml"
hyp: str = "/data/sen.li/workspace/code/FlagPerf/training/benchmarks/yolov5/pytorch/dataloaders/hyp.scratch-low.yaml"

resume: bool = False
data: str = "coco.yaml"
# cuda device, i.e. 0 or 0,1,2,3 or cpu
device: str = "0"
# initial weights path
weight: str = "yolov5s.pt"

epoch: int = 300

data_dir = "/data/sen.li/workspace/datasets/yolov5/coco/images"
train_data = "train2017"
eval_data = "val2017"
imgsz = 640
batch_size = 64

data: str = "/data/coco.yaml"
gs = 32 # grid size (max stride)
single_cls = True
pad = 0.5
# hyp is path/to/hyp.yaml or hyp dictionary
hyp = "/data/sen.li/workspace/code/yolov5/data/coco.yaml"
augment = True
cache = False
rect = True

rank = -1
workers = 4
image_weights = True
quad = False
prefix = ''
shuffle = True