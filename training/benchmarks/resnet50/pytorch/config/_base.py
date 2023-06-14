# chip vendor, required
vendor: str = None

# random seed
seed: int = 1234
# torch.backends.cudnn.benchmark
cudnn_benchmark: bool = False
# torch.backends.cudnn.deterministic
cudnn_deterministic: bool = True
# model args
name: str = "resnet50"

# refer to torchvision official site for recommended configs
# https://github.com/pytorch/vision/tree/main/references/classification

# lr_scheduler args
# initial learning rate
learning_rate: float = 0.1
lr_step_size: int = 30
lr_gamma: float = 0.1

# learning rate decay function
lr_decay_style: str = "linear"

# optimizer args
# weight decay coefficient for L2 regularization
weight_decay_rate: float = 1e-4
# momentum for SGD optimizer
momentum: float = 0.9

# fp16 config args
fp16: bool = False

# Static loss scaling, positive power of 2 values can improve fp16 convergence. If None, dynamicloss scaling is used.
loss_scale: float = 4096

# Window over which to raise/lower dynamic scale
loss_scale_window: float = 1000

# Minimum loss scale for dynamic loss scale
min_scale: float = 1

# distributed args

# load and save args
# Path to a directory containing a model checkpoint.
init_checkpoint: str = None

# data args
# Training data dir
data_dir: str = None

# Number of workers to use for dataloading
num_workers: int = 2

# Total batch size for training.
train_batch_size: int = 128

# Total batch size for validating.
eval_batch_size: int = 128

# Maximum sequence length to process
seq_length: int = 200

# trainer args
do_train: bool = True

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: int = 1

# Total number of training samples to run.
max_samples_termination: int = 0

# number of training samples to run a evaluation once
eval_interval_samples: int = 20000

# Sample to begin performing eval.
eval_iter_start_samples: int = 1

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 1

# target accuracy to converge for training
target_acc1: float = 75.5
# dist args
# Whether to read local rank from ENVVAR
use_env: bool = True

# local_rank for distributed training on gpus or other accelerators
local_rank: int = -1

# Communication backend for distributed training on gpus
dist_backend: str = "nccl"

# Distributed Data Parallel type
ddp_type: str = "native"

# device
device: str = None
n_device: int = 1

distributed: bool = False
pretrained: bool = False

gpu: int = None
multiprocessing_distributed: bool = False
print_freq: int = 10

num_epochs_to_generate_seeds_for: int = 2
