# =========================================================
# Required parameters
# =========================================================
vendor: str = None

device: str = "gpu"


# =========================================================
# data
# =========================================================
# vocab file path
tokenizer_vocab_file : str = 'sentencepiece.bpe.model'

# The name of the dataset to use (via the datasets library).
input_dir : str = "data" 

# Train/valid/test data split.
split: str = "949,50,1"

# The maximum total input sequence length after tokenization. Sequences longer 
max_seq_length: int = 2048

# Use share folder for data dir and output dir on multi machine.
share_folder: bool = False

dataset_rank: int = 0

dataloader_num_workers: int = 1


# =========================================================
# Model
# =========================================================
# Only support for llama pre-training for now.
model_type: str = "llama"

model_name_or_path: str = "facebook/llama-7b"

hidden_size: int = 4096 # 4096, 768

initializer_range: float = 0.02

intermediate_size: int = 11008

lm_shift_labels: bool = False

max_position_embeddings: int = 2048

num_attention_heads: int = 32 # 32, 8

num_hidden_layers: int = 32 # 32, 2

rms_norm_eps: float = 1e-06

vocab_size: int = 32000

bos_token_id: int = 1

eos_token_id: int = 2

pad_token_id: int = 0

use_cache: bool = False

recompute: bool = True

tensor_parallel_output: bool = True

tie_word_embeddings: bool = False

use_flash_attention: bool = False

# Pretrained tokenizer name or path if not the same as model_name
tokenizer_name_or_path: str = "facebook/llama-7b"

# llama, use_fused_rms_norm
use_fused_rms_norm: bool = False

# gpt, fuse_attention_qkv
fuse_attention_qkv: bool = True

fuse_attention_ffn: bool = False

# full core_attn
recompute_granularity: str = "full"

# Pre-training from existing paddlenlp model weights. Default Fasle and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models.
continue_training: bool = True

num_workers: int = 1 

dataloader_drop_last: bool = False

dataset_world_size: int = 1


# =========================================================
# trainer args
# =========================================================
# Do trainingFalse
do_train: bool = True

do_eval: bool = True

# Total number of training steps to perform.
max_steps: int = 10000

per_device_train_batch_size: int = 1

per_device_eval_batch_size: int = 1

# Total number of training samples to run.
max_samples_termination: float = 120000

# frequency of logging loss. If not positive, no logging is provided for training loss
logging_steps: int = 20

log_freq = logging_steps

logging_dir: str = None

eval_steps: int = 1000

# Sample to begin performing eval.
eval_iter_start_samples: int = 1

eval_iters: int = 10

test_iters = eval_iters * 10

# The steps use to control the learing rate. If the step > decay_steps, will use the min_learning_rate.
decay_steps: float = None

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps : int = 1

local_rank : int = -1

local_process_index : int = 0

# random seed
seed: int = 42

world_size : int = 1

max_grad_norm: float = 1.0

use_hybrid_parallel: bool = True

sharding: str = "stage2"

disable_tqdm : bool = True

# =========================================================
# fp16 config args
# =========================================================
# Run model in fp16 mode
fp16: bool = True

fp16_opt_level: str = 'O2'

bf16: bool = False

scale_loss: float = 1024.0

amp_custom_white_list = None

amp_custom_black_list = None


# =========================================================
# dist args
# =========================================================
# Whether to read local rank from ENVVAR
use_env: bool = True

# Communication backend for distributed training on gpus
dist_backend: str = "nccl"


# =========================================================
# lr_scheduler args
# =========================================================
# initial learning rate
learning_rate: float = 0.0001

# Minimum learning rate deacyed to.
min_learning_rate : float = 1e-05

# number of iterations to decay LR over, If None defaults to `--train-iters`*`--epochs`
lr_decay_steps: int = 10

# learning rate decay function
lr_scheduler_type: str = "linear"

# percentage of data to warmup on (.01 = 1% of all training iters). Default 0.01
warmup_ratio: float = 0.01

warmup_steps: int = 0

# weight decay coefficient for L2 regularization
weight_decay: float = 0.01


# =========================================================
# optimizer args
# =========================================================
adam_beta1: float = 0.9
adam_beta2: float = 0.999
adam_epsilon: float = 1e-08


# =========================================================
# load and save args
# =========================================================
# Path to a directory containing a model checkpoint.
init_checkpoint: str = "model_state.pdparams"
output_dir: str = "llama-paddle/output"