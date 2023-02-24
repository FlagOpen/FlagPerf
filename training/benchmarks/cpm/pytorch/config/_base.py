from typing import ClassVar
# from train.event.base import BaseTrainingEventInterface

# random seed
seed: int = 1234

## model args
# num decoder layers
num_layers: int = 24

# transformer hidden size
hidden_size: int = 1024

# num of transformer attention heads
num_attention_heads: int = 16

# vocab size to use for non-character-level tokenization. This value will only be used when creating a tokenizer
vocab_size: int = 30000

# dropout probability for hidden state transformer
hidden_dropout: float = 0.5

# maximum number of position embeddings to use
max_position_embeddings: int = 1024

# dropout probability for attention weights
attention_dropout: float = 0.5

layernorm_epsilon: float = 1.0e-5

## lr_scheduler args
# initial learning rate
learning_rate: float = 0.0001

# number of iterations to decay LR over, If None defaults to `--train-iters`*`--epochs`
lr_decay_iters: int = None

# learning rate decay function
lr_decay_style: str = "linear"

# percentage of data to warmup on (.01 = 1% of all training iters). Default 0.01
warmup: float = 0.01

warmup_steps: int = 0

## optimizer args
# weight decay coefficient for L2 regularization
weight_decay_rate: float = 0.05

beta_1: float = 0.9
beta_2: float = 0.99
eps: float = 1e-08

##fp16 config args
# Run model in fp16 mode
fp16: bool = True

# Static loss scaling, positive power of 2 values can improve fp16 convergence. If None, dynamicloss scaling is used.
loss_scale: float = 4096

dynamic_loss_scale: bool = True

# hysteresis for dynamic loss scaling
hysteresis: int = 2

# Window over which to raise/lower dynamic scale
loss_scale_window: float = 1000

# Minimum loss scale for dynamic loss scale
min_scale: float = 1

## distributed args
# Turn ON gradient_as_bucket_view optimization in native DDP.
use_gradient_as_bucket_view: bool = False

## load and save args
# Path to a directory containing a model checkpoint.
init_checkpoint = "cpm_model_states_medium.pt"

# Output directory to save checkpoints to.
save: str = None

# number of iterations between saves
save_interval: int = 5000

# Do not save current optimizer.
no_save_optim: bool = False

# Do not save current rng state.
no_save_rng: bool = False

## data args
# Training data dir
data_dir: str = "/mnt/data/cpm/train/"

# path used to save/load sentencepiece tokenization models
tokenizer_path: str = "bpe_3w_new/"
tokenizer_vocab_file = 'vocab.json'
tokenizer_vocab_model = 'chinese_vocab.model'

# Number of workers to use for dataloading
num_workers: int = 2

# Total batch size for training.
train_batch_size: int = 32

# Total batch size for training.
eval_batch_size: int = 32

# Maximum sequence length to process
seq_length: int = 200

## trainer args
# Do trainingFalse
do_train: bool = True

# total number of iterations to train over all training runs
epoch: int = 10

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: int = 1

# checkpoint activation to allow for training with larger models and sequences
checkpoint_activations: bool = True

# chunk size (number of layers) for checkpointing
checkpoint_num_layers: int = 1

# Total number of training steps to perform.
max_steps: int = 10000

# Stop training after reaching this Embedding_average
target_embedding_average: float = 0.92

# Total number of training samples to run.
max_samples_termination: float = 43912600

eval_interval_samples: int = 20000

# Sample to begin performing eval.
eval_iter_start_samples: int = 1

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 1

## dist args
# Whether to read local rank from ENVVAR
use_env: bool = True

# local_rank for distributed training on gpus or other accelerators
local_rank: int = -1

# Communication backend for distributed training on gpus
dist_backend: str = "nccl"

ddp_type: str = "native"

# device
device: str = None
n_device: int = 1

# A object to provide some core components in training
# training_event: ClassVar[BaseTrainingEventInterface] = None

# training_event_instance: BaseTrainingEventInterface = None
