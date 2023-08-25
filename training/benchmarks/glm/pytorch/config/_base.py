# required parameters

# case info
# chip vendor: nvidia, kunlunxin, iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
name: str = "GLM"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True
data_dir: str = None

do_train = True
fp16 = True
# =========================================================
# data
# =========================================================

train_data: str = "ReCoRD/glm_train_eval_hdf5_sparse/train_hdf5/train_sparse.hdf5"
eval_data: str = "ReCoRD/glm_train_eval_hdf5_sparse/eval_hdf5/eval_sparse.hdf5"
output_dir: str = ""
init_checkpoint: str = "blocklm-large-blank/200000/mp_rank_00_model_states.pt"

# =========================================================
# Model
# =========================================================
max_seq_length: int = 512
num_layers: int = 24
hidden_size: int = 1024
num_attention_heads: int = 16
hidden_dropout: float = 0.1
attention_dropout: float = 0.1
max_position_embeddings: int = 512
mem_length: int = 0
checkpoint_num_layers: int = 1
attention_scale: float = 1
vocab_size: int = 30592
checkpoint_activations = True
max_memory_length = 0

# =========================================================
# loss scale
# =========================================================
loss_scale = None
dynamic_loss_scale: bool = True
loss_scale_window: int = 1000
min_scale: int = 1
hysteresis: int = 2

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8
dist_backend: str = 'nccl'

lr: float = 1e-5
weight_decay: float = 0.1
adam_beta1: float = 0.9
adam_beta2: float = 0.999
adam_eps: float = 1e-08
lr_decay_iters: int = 4338
gradient_accumulation_steps: int = 1
warmup: float = 0.1
lr_decay_ratio: float = 0.1

clip_grad: float = 1.0

seed: int = 10483

# Stop training after reaching this accuracy
target_accuracy: float = 0.8

# Sample to begin performing eval.
eval_iter_start_samples: int = 100

# If set to -1, disable eval, else evaluate every eval_iter_samples during training
eval_interval_samples: int = 600 * 8 * 4

# Total number of training samples to run.
max_samples_termination: float = 1388270 * 4

# number workers for dataloader
num_workers: int = 4

# local_rank for distributed training on gpus
local_rank: int = -1
# Whether to read local rank from ENVVAR
use_env: bool = True

# Number of epochs to plan seeds for. Same set across all workers.
num_epochs_to_generate_seeds_for: int = 2

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 1

# Whether to resume training from checkpoint.
# If set, precedes init_checkpoint/init_tf_checkpoint
resume_from_checkpoint: bool = False

# A object to provide some core components in training
#training_event: ClassVar[BaseTrainingEventInterface] = None

#training_event_instance: BaseTrainingEventInterface = None

# device
device: str = None
n_device: int = 1