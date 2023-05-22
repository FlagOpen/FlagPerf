from typing import ClassVar
#from train.event.base import BaseTrainingEventInterface

# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
# vendor: str = None
vendor: str = "iluvatar"
# model name
name: str = "yolov5"

do_train = True
fp16 = False
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
target_mAP_5: float = 50.0

# Sample to begin performing eval.
eval_iter_start_samples: int = 100

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_interval_samples: int = 100 * 256 * 1

# Total number of training samples to run.
max_samples_termination: float = 1388270 * 4 * 30

# number workers for dataloader
num_workers: int = 8

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

cfg: str = "/workspace/flagperf/training/benchmarks/yolov5/pytorch/models/yolov5s.yaml"
hyp: str = "/workspace/flagperf/training/benchmarks/yolov5/pytorch/dataloaders/hyp.scratch-low.yaml"
# initial weights path
weights: str = "/workspace/flagperf/training/benchmarks/yolov5/pytorch/models/yolov5s.pt"
data: str = "/workspace/flagperf/training/benchmarks/yolov5/pytorch/dataloaders/coco.yaml"
save_dir: str = "exp"

resume: bool = False
# cuda device, i.e. 0 or 0,1,2,3 or cpu
device: str = "0,1"
n_device: int = 2
epochs: int = 100

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
#path: ../datasets/coco  # dataset root dir
data_dir: str = "/mnt/data/yolov5/train"
train_data: str = "train2017.txt"  # train images (relative to 'path') 118287 images
eval_data:str = "val2017.txt"  # val images (relative to 'path') 5000 images
test_data:str = "test-dev2017.txt"  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
nc: 80  # number of classes
class_names: list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

imgsz = 640
batch_size = 16
gs = 32 # grid size (max stride)
pad = 0.5
augment = True
cache = None
rect = False

rank = -1
workers = 4
image_weights = False
quad = False
prefix = ''
shuffle = True

multi_scale=False, 
single_cls=False
sync_bn=False
optimizer: str = "SGD"
cos_lr = False
start_epoch=False
label_smoothing=0.0
patience=100
freeze=[0]
save_period=-1
seed=0
# local_rank=-1
# local_rank: int = 0
entity=None
upload_dataset=False
bbox_interval=-1
artifact_alias="latest"
noval=False

distributed: bool = True
