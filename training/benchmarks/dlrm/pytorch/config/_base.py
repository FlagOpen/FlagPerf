# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = "nvidia"
data_dir: str = None
name: str = "dlrm"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 100
print_freq: int = 200
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1

# Optional parameters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "validation"

# Name of the feature spec file in the dataset directory
feature_spec: str = "feature_spec.yaml"
# valid options: 'synthetic_gpu', 'parametric'
dataset_type: str = "parametric"
# Read batch in train dataset by random order 
shuffle_batch_order: bool = False
# Maximum number of rows per embedding table, by default equal to the number of unique values for each categorical variable
max_table_size: int = None
# If True the model will compute `index := index % table size`, to ensure that the indices match table sizes
hash_indices: bool = False

# =========================================================
# Synthetic data
# =========================================================
# Number of samples per epoch for the synthetic dataset
synthetic_dataset_num_entries: int = 2**15 * 1024
# Cardinalities of variables to use with the synthetic dataset.
synthetic_dataset_table_sizes: str = ','.join(26 * [str(10**5)])
# Number of numerical features to use with the synthetic dataset
synthetic_dataset_numerical_features: int = 13
# Create a temporary synthetic dataset based on a real one.  Uses --dataset and --feature_spec Overrides synthetic_dataset_table_sizes and synthetic_dataset_numerical_features.
# --synthetic_dataset_num_entries is still required.
synthetic_dataset_use_feature_spec: bool = False

# =========================================================
# Learning rate schedule
# =========================================================
# Base learning rate
lr: float = 24
# Learning rate warmup factor. Must be a non-negative integer
warmup_factor: int = 0
# Number of warmup optimization steps
warmup_steps: int = 8000
# Polynomial learning rate decay steps. If equal to 0 will not do any decaying
decay_steps: int = 24000
# Optimization step after which to start decaying the learning rate, if None will start decaying right after the warmup phase is completed
decay_start_step: int = 48000
# Polynomial learning rate decay power
decay_power: int = 2
# LR after the decay ends
decay_end_lr: float = 0

# =========================================================
# train && evaluate
# =========================================================
# Batch size used for training
train_batch_size: int = 0
# Batch size used for testing/validation
eval_batch_size: int = 0

# refer: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM#training-accuracy-nvidia-dgx-a100-8x-a100-80gb
target_auc: float = 0.802
# Number of epochs to train for
max_epoch: int = 1
# Stop training after doing this many optimization steps
max_steps: int = None

# =========================================================
# Checkpointing
# =========================================================
# Path from which to load a checkpoint
load_checkpoint_path: str = None
# Path to which to save the training checkpoints
save_checkpoint_path: str = None

# =========================================================
# Saving and logging
# =========================================================
# Destination for the log file with various results and statistics
log_path: str = "./log.json"
# Number of optimization steps between validations. If None will test after each epoch"
test_freq: int = None
# Don't test the model unless this many epochs has been completed
test_after: float = 0

# Number of initial iterations to exclude from throughput measurements
benchmark_warmup_steps: int = 0

# =========================================================
# Machine
# =========================================================
# Device to run the majority of the model operations
base_device = "cuda"
# If True the script will use Automatic Mixed Precision
amp = True
# Use CUDA Graphs
cuda_graphs = True

# =========================================================
# utils
# =========================================================
seed: int = 0
do_train = True
fp16 = False
distributed: bool = True
num_workers: int = 4
device: str = None

# =========================================================
# Miscellaneous
# =========================================================
dist_backend: str = 'nccl'

# Use an optimized implementation of MLP from apex
optimized_mlp: bool = True

device: str = None
# Specifies where ROC AUC metric is calculated
auc_device: str = "GPU"
# Sort features from the bottom model, useful when using saved
bottom_features_ordered: bool = False

# For debug and benchmarking. Don't perform the weight update for MLPs.
freeze_mlps: bool = False
# For debug and benchmarking. Don't perform the weight update for the embeddings.
freeze_embeddings: bool = False
# Swaps embedding optimizer to Adam
Adam_embedding_optimizer: bool = False
# Swaps MLP optimizer to Adam
Adam_MLP_optimizer: bool = False
