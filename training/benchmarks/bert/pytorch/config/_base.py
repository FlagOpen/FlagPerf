from typing import ClassVar

# chip vendor: nvidia, kunlunxin, iluvatar, cambricon etc. key vendor is required.
vendor: str = None

# The train dir. Should contain train_dir, eval_dir, init_checkpoint, bert_config_path for the task.
data_dir: str = None

# The train dir. Should contain .hdf5 files  for the task.
train_dir: str = None

# Bert pre-trained model selected in the list:
# bert-base-uncased, bert-large-uncased, bert-base-cased,
# bert-base-multilingual, bert-base-chinese.
bert_model: str = "bert-large-uncased"

# The output directory where the model checkpoints will be written.
output_dir: str = None

# The eval data dir. Should contain .hdf5 files  for the task.
eval_dir: str = None

# Sample to begin performing eval.
eval_iter_start_samples: int = 0

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_iter_samples: int = 40000

eval_step: int = 2000

# number of eval examples to run eval on
num_eval_examples: int = 10000

# whether to cache evaluation data on GPU
cache_eval_data: bool = False

# The initial checkpoint to start training from.
init_checkpoint: str = None

# The initial TF checkpoint to start training from.
init_tf_checkpoint: str = None

# Whether to verify init checkpoint.
verify_checkpoint: bool = True

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter
# than this will be padded.
max_seq_length: int = 512

# The maximum total of masked tokens in input sequence
max_predictions_per_seq: int = 76

# Total batch size for training.
train_batch_size: int = 18

# Total batch size for training.
eval_batch_size: int = 128

# The initial learning rate for LAMB.
learning_rate: float = 4e-05

# weight decay rate for LAMB.
weight_decay_rate: float = 0.01

# LAMB beta1.
opt_lamb_beta_1: float = 0.9

# LAMB beta2.
opt_lamb_beta_2: float = 0.999

# Total number of training steps to perform.
max_steps: int = 1536

# Total number of training samples to run.
max_samples_termination: float = 14000000

# Proportion of optimizer update steps to perform linear learning rate warmup for.
# Typically 1/8th of steps for Phase2
warmup_proportion: float = 0.01

# Number of optimizer update steps to perform linear learning rate warmup for.
# Typically 1/8th of steps for Phase2
warmup_steps: int = 0

# Starting step for warmup.
start_warmup_step: int = 0

# local_rank for distributed training on gpus
local_rank: int = -1

# Communication backend for distributed training on gpus
dist_backend: str = "nccl"
# random seed for initialization
seed: int = 42

# Number of updates steps to accumualte before performing a backward/update pass.
gradient_accumulation_steps: int = 1

# Whether to use 16-bit float precision instead of 32-bit
fp16: bool = False

# Loss scaling, positive power of 2 values can improve fp16 convergence.
loss_scale: float = 0.0

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 100

# Whether to use gradient checkpointing
checkpoint_activations: bool = False

# Whether to resume training from checkpoint.
# If set, precedes init_checkpoint/init_tf_checkpoint
resume_from_checkpoint: bool = False

# The initial checkpoint to start continue training from.
resume_init_checkpoint: str = None

# Number of checkpoints to keep (rolling basis).
keep_n_most_recent_checkpoints: int = 20

# Number of update steps until a model checkpoint is saved to disk.
num_samples_per_checkpoint: int = 500000

# Number of update steps until model checkpoints start saving to disk.
min_samples_to_start_checkpoints: int = 3000000

# Whether to save checkpoints
save_checkpoint: bool = False

# Whether to run training.
do_train: bool = True

# Whether to run with unpadding.
exchange_padding: bool = False

# Whether to disable fusion of attention mask to softmax and dropout.
enable_fuse_dropout: bool = False

# Whether to disable fusion of the attention mask to softmax.
disable_fuse_mask: bool = False

# Whether to run with optimizations.
fused_gelu_bias: bool = False

use_xpu: bool = False
# Whether to run with optimizations.
fused_dropout_add: bool = False

# Whether to run with optimizations.
dense_seq_output: bool = False

# Whether to read local rank from ENVVAR
use_env: bool = True

# Path bert_config.json is located in
bert_config_path: str = None

# Stop training after reaching this Masked-LM accuracy
target_mlm_accuracy: float = 0.710

# Average accuracy over this amount of batches before performing a stopping criterion test
train_mlm_accuracy_window_size: int = 0

# Number of epochs to plan seeds for. Same set across all workers.
num_epochs_to_generate_seeds_for: int = 2

# Enable DDP.
use_ddp: bool = False

# Turn ON gradient_as_bucket_view optimization in native DDP.
use_gradient_as_bucket_view: bool = False

# device
device: str = None
n_device: int = 1
n_gpu: int = 1

eval_interval_samples: int = 0
