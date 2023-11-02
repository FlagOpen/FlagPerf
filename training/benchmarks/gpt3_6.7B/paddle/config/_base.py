# =========================================================
# Required parameters
# =========================================================
vendor: str = None

device = "gpu"

# =========================================================
# data
# =========================================================
# The name of the dataset to use (via the datasets library).
input_dir: str = "data"

# Train/valid/test data split.
split: str = "949,50,1"

# The maximum total input sequence length after tokenization. Sequences longer
max_seq_length: int = 2048

# Use share folder for data dir and output dir on multi machine.
share_folder: bool = False


# =========================================================
# Model
# =========================================================
# Only support for llama pre-training for now.
model_type = "gpt"

model_name_or_path = "gpt3-6.7B-en"

tokenizer_name_or_path = "gpt3-6.7B-en"

hidden_size = 4096  # for gpt3-6.7b, 5120 for gpt3-13b

initializer_range = 0.02

intermediate_size = 16384  # for gpt3-6.7b, 20480 for gpt3-13b

lm_shift_labels = False

max_position_embeddings = 2048

num_attention_heads = 32  # for gpt3-6.7b, 128 for gpt3-13b

num_hidden_layers = 32  # for gpt3-6.7b, 40 for gpt3-13b

hidden_act = "gelu"

hidden_dropout_prob = 0.1

attention_probs_dropout_prob = 0.1

rms_norm_eps = 1e-06

vocab_size = 50304

bos_token_id = 1

eos_token_id = 50256

eol_token_id = 198

pad_token_id = 0

use_cache = False

tensor_parallel_output = True

tie_word_embeddings = False

# Pretrained tokenizer name or path if not the same as model_name
tokenizer_name_or_path = None

# virtual_pp_degree
virtual_pp_degree: int = 1

# Pre-training from existing paddlenlp model weights. 
# Default Fasle and model will train from scratch. 
# If set True, the model_name_or_path argument must exist in the paddlenlp models.
continue_training: bool = False

num_workers: int = 1

dataloader_drop_last: bool = False


# =========================================================
# trainer args
# =========================================================
# Do trainingFalse
do_train: bool = True

do_eval: bool = True

do_predict: bool = True

target_ppl = 10000

# Total number of training steps to perform.
max_steps: int = 512*1024*1024

save_steps: int = 10000

per_device_train_batch_size = 1

per_device_eval_batch_size = 1

dataloader_num_workers = 1

# Total number of training samples to run.
max_samples_termination: float = 120000

# frequency of logging loss. If not positive, no logging is provided for training loss
logging_steps: int = 20

log_freq = logging_steps

logging_dir: str = None

eval_steps = 5000000

# Sample to begin performing eval.
eval_iter_start_samples: int = 1

eval_iters = 10

test_iters = eval_iters * 10

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: int = 2

local_rank: int = -1
# local_process_index: int = 0

target_loss: float = 7.0

# random seed
seed: int = 42

max_grad_norm: float = 1.0

# =========================================================
# parallel args
# =========================================================

recompute = False

use_flash_attention: bool = False

# 需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP
use_fused_rms_norm: bool = False

# fuse_attention_qkv
fuse_attention_qkv: bool = False

fuse_attention_ffn: bool = False

# full core_attn
recompute_granularity: str = "full"

sharding: str = "stage2"

sharding_degree: int = -1

tensor_parallel_degree = 1

pipeline_parallel_degree = 1

distributed_dataloader = True

# =========================================================
# fp16 config args
# =========================================================
# Run model in fp16 mode
fp16: bool = True

fp16_opt_level = "O2"

bf16: bool = False

scale_loss = 32768.0

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
learning_rate: float = 1.2e-4 # for gpt3-6.7B, 1.0 for gpt3-13B

# Minimum learning rate deacyed to.
min_learning_rate: float = learning_rate * 0.1

# learning rate decay function
lr_scheduler_type: str = "cosine"

# Initial learning rate decay per epoch
# decay tokens = 260B, batch_size tokens= 2M, 260B / 2M = 130M
lr_decay_steps: int = 130*1024*1024
decay_steps: float = 130*1024*1024


# warmup tokens = 375M, batch_size tokens= 2M, 375M / 2M = 187.5 steps
warmup_steps: int = 188
# warmup_ratio: float = 0.01

# weight decay coefficient for L2 regularization
weight_decay: float = 0.1

# =========================================================
# optimizer args
# =========================================================
adam_beta1: float = 0.9
adam_beta2: float = 0.95
adam_epsilon: float = 1e-08


# =========================================================
# load and save args
# =========================================================
# Path to a directory containing a model checkpoint.
init_checkpoint = "model_state.pdparams"

output_dir = "gpt3-paddle/output"

disable_tqdm = True
