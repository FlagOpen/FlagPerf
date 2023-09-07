# =========================================================
# Required parameters
# =========================================================
vendor: str = None

device: str = "gpu"


# =========================================================
# data
# =========================================================
# The name of the dataset to use (via the datasets library).
input_dir : str = "data" 

# Train/valid/test data split.
split: str = "949,50,1"

# The maximum total input sequence length after tokenization. Sequences longer "
# "than this will be truncated, sequences shorter will be padded.
max_seq_length: int = 2048

# Mask token prob.
masked_lm_prob: float = 0.15

# Short sequence prob.
short_seq_prob: float = 0.

# Use share folder for data dir and output dir on multi machine.
share_folder: bool = False

# Whether to favor long ngrams
favor_longer_ngram: bool = False

# Max N Grams
max_ngrams: int = 3

# mmap/lazy format converted from preprocessed data.
data_impl: str = "mmap"

# Drop the last incomplete batch if it is not divisible by the batch size.
dataloader_drop_last: bool = False

# Number of subprocesses to use for data loading. 
# 0 means that the data will be loaded in the main process.
dataloader_num_workers: int = 1


# =========================================================
# Model
# =========================================================
# Only support for llama pre-training for now.
model_type: str = "llama"

# Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html
model_name_or_path: str = "facebook/llama-7b" # "facebook/llama-7b"

# Pretrained tokenizer name or path if not the same as model_name
tokenizer_name_or_path: str = "facebook/llama-7b"

# Pre-training from existing paddlenlp model weights. Default Fasle and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models.
continue_training: bool = True

# use flash attention
use_flash_attention: bool = False

# use fused rms_norm
use_fused_rms_norm: bool = False

# =========================================================
# trainer args
# =========================================================
# The output directory where the model predictions and checkpoints will be written.
output_dir: str = None

# Whether to run training.
do_train: bool = True

# Whether to run eval on the dev set.
do_eval: bool = True

# Batch size per GPU core/CPU for training.
per_device_train_batch_size: int = 1

# Batch size per GPU core/CPU for evaluation.
per_device_eval_batch_size: int = 1

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: int = 1

# If > 0: set total number of training steps to perform. Override num_train_epochs.
max_steps: int = -1

# Log every X updates steps.
logging_steps: int = 20
log_freq = logging_steps

# Random seed that will be set at the beginning of training.
seed: int = 42

# Whether or not to use Paddle Sharding Data Parallel training (in distributed training
# only). The base option should be `stage1`, `stage2` or `stage3` and you can add
# CPU-offload to `stage2` or `stage3` like this: stage2 offload` or `stage3 offload`. 
# sharding: str = None

# tensor_parallel_degree means split the transformer layer to how many parts.
# default -1 for not use tensor parallel,  Suggest tensor_parallel_degree<=8 for better proformance.
# Note, this need model support in source code.
tensor_parallel_degree: int = -1

# pipeline_parallel_degree means split all transformer layers to how many stages.
# default -1 for not use pipeline parallel.
# Note. this need model support in source code, see llama modeling_pp.py file
pipeline_parallel_degree: int = -1

# Recompute the forward pass to calculate gradients. Used for saving memory.
recompute: bool = True

# Whether or not to disable the tqdm progress bars.
disable_tqdm : bool = True

# Run an evaluation every X steps.
eval_steps: int = 1000

# Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
save_steps: int = 5000

# The steps use to control the learing rate. If the step > decay_steps, will use the min_lr.
decay_steps: int = None

# virtual_pp_degree
virtual_pp_degree: int = 1

# use sequence parallel. If mp_degree=1, sequence_parallel is forced to be False.
sequence_parallel: bool = False

# Whether to use distributed dataloader
distributed_dataloader: bool = True

# recompute训练的粒度
# 可选 `full` `full_attn` `core_attn`
# full即recompute全部transformer
# full_attn表明只recompute所有self attention部分
# core_attn表明只recompute `softmax(qkT)v` 部分
# 注：显存占用方面，`core_attn` > `full_attn` > `full`，若所选策略产生OOM错误，可以适当更改
recompute_granularity: int = "full"

# target perplexity value
target_ppl: float = 10.0

# =========================================================
# fp16 config args
# =========================================================
# Whether to use fp16 (mixed) precision instead of 32-bit
fp16: bool = True

# For fp16: AMP optimization level selected in ['O0', 'O1', and 'O2']. 
fp16_opt_level: str = 'O0'

# Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA
# architecture or using CPU (no_cuda). This is an experimental API and it may change.
bf16: bool = False

# The value of initial scale_loss for fp16.
scale_loss: float = 1024.0


# =========================================================
# dist args
# =========================================================
# Whether to read local rank from ENVVAR
use_env: bool = True

# Communication backend for distributed training on gpus
dist_backend: str = "nccl"

local_rank: int = -1


# =========================================================
# lr_scheduler args
# =========================================================
# initial learning rate
learning_rate: float = 0.0001

# Minimum learning rate deacyed to.
min_learning_rate : float = 1e-05

# Linear warmup over warmup_ratio fraction of total steps.
warmup_ratio: float = 0.01

# Linear warmup over warmup_steps.
warmup_steps: int = 0

# weight decay coefficient for L2 regularization
weight_decay: float = 0.01

# The scheduler type to use. suppor linear, cosine, constant, constant_with_warmup
lr_scheduler_type: str = "linear"


# =========================================================
# optimizer args
# =========================================================
# Beta1 for AdamW optimizer
adam_beta1: float = 0.9

# Beta2 for AdamW optimizer
adam_beta2: float = 0.999

# Epsilon for AdamW optimizer.
adam_epsilon: float = 1e-8

# Max gradient norm.
max_grad_norm: float = 1.0


# =========================================================
# load and save args
# =========================================================
# Path to a directory containing a model checkpoint.
output_dir: str = "llama-paddle/output"