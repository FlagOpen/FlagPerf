# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "moflow"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# data
# =========================================================
# The config to choose. This parameter allows one to switch between different datasets.
# and their dedicated configurations of the neural network. By default, a pre-defined "zinc250k" config is used.
dataset_name: str = "zinc250k"
# Number of workers in the data loader.
num_workers: int = 4

# =========================================================
# loss scale
# =========================================================
# Base learning rate.
lr: float = 0.0005
# beta1 parameter for the optimizer.
beta1: float = 0.9
# beta2 parameter for the optimizer.
beta2: float = 0.99
# Gradient clipping norm.
clip: float = 1.0
# =========================================================
# train && evaluate
# =========================================================
# Batch size per GPU for training
train_batch_size: int = 512
eval_batch_size: int = 100

target_nuv: float = 80

# Frequency for saving checkpoints, expressed in epochs. If -1 is provided, checkpoints will not be saved.
save_epochs: int = 5
# Evaluation frequency, expressed in epochs. If -1 is provided, an evaluation will not be performed.
eval_epochs: int = 5

# Number of warmup steps. This value is used for benchmarking and for CUDA graph capture.
warmup_steps: int = 20
# Number of steps used for training/inference. This parameter allows finishing.
# training earlier than the specified number of epochs.
# If used with inference, it allows generating  more molecules (by default only a single batch of molecules is generated).
steps: int = -1
# Temperature used for sampling.
temperature: float = 0.3
first_epoch: int = 0
epochs: int = 300

allow_untrained = False

do_train = True
fp16 = False
amp: bool = True
distributed: bool = True

# Directory where checkpoints are stored
results_dir: str = "moflow_results"
# Path to store generated molecules. If an empty string is provided, predictions will not be saved (useful for benchmarking and debugging).
# predictions_path: str = "moflow_results/predictions.smi"
# =========================================================
# experiment
# =========================================================
# Compile the model with `torch.jit.script`. Can be used to speed up training or inference.
jit: bool = False
# Capture GPU kernels with CUDA graphs. This option allows to speed up training
cuda_graph: bool = True
# Verbosity level. Specify the following values: 0, 1, 2, 3, where 0 means minimal verbosity (errors only) and 3 - maximal (debugging).
verbosity: int = 1
# Path for DLLogger log. This file will contain information about the speed and accuracy of the model during training and inference.
# Note that if the file already exists, new logs will be added at the end.
log_path: str = "moflow_results/moflow.json"
# Frequency for writing logs, expressed in steps.
log_interval: int = 20


# Apply validity correction after the generation of the molecules.
correct_validity: bool = False
# =========================================================
# utils
# =========================================================
# Random seed used to initialize the distributed loaders
seed: int = 1
dist_backend: str = 'nccl'

device: str = None

# =========================================================
# for driver
# =========================================================
# rank of the GPU, used to launch distributed training.
local_rank: int = -1
use_env: bool = True
log_freq: int = 500
print_freq: int = 500
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1
