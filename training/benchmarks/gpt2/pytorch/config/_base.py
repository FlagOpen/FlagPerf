# Required parameters

vendor: str = None
data_dir: str = None
name: str = "GPT2"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

use_env: bool = True
log_freq: int = 1
device: str = None

# =========================================================
# train config
# =========================================================

seed: int = 1234
gradient_accumulation_steps: int = 1

max_steps: int = 23070
train_batch_size: int = 4

eval_iter_start_samples: int = 3200
eval_interval_samples: int = 3200

target_acc: float = 0.60

# =========================================================
# data
# =========================================================

train_data_prefix: str = "lambada_train_text_document"
test_data_prefix: str = "lambada_test.json"
vocab_file: str = "gpt2-vocab.json"
merge_file: str = "gpt2-merges.txt"

# =========================================================
# loss scale
# =========================================================
clip_grad: float = 1.0

# =========================================================
# optimizer & lr scheduler & weight decay
# =========================================================
optimizer: str = "adam"
adam_beta1: float = 0.9
adam_beta2: float = 0.999
adam_eps: float = 1e-8

lr: float = 0.00015
min_lr: float = 1e-05
lr_warmup_fraction: float = 0.01
lr_warmup_iters: int = 0
lr_warmup_samples: int = 0
lr_decay_style: str = "cosine"
lr_decay_samples: int=None

weight_decay: float = 0.01
start_weight_decay: float = 0.01
end_weight_decay: float = 0.01
weight_decay_incr_style: str = "constant"

use_distributed_optimizer: bool = False
barrier_with_L1_time: bool = True

# =========================================================
# transformer
# =========================================================

num_layers: int = 24
encoder_num_layers: str = 24

num_attention_heads: int = 16
hidden_size: int = 1024
ffn_hidden_size: int = 4096
kv_channels: int = 64
seq_length: int = 1024
attention_dropout: float = 0.1
hidden_dropout: float = 0.1
transformer_impl: str = "local"
use_flash_attn: bool = False

layernorm_epsilon: float = 1e-05

fp16: bool = False
bf16: bool = False

init_method_std: float = 0.02
import torch
params_dtype = torch.float32
masked_softmax_fusion: bool = True
bias_gelu_fusion: bool = True
bias_dropout_fusion: bool = True
apply_residual_connection_post_layernorm: bool = False
apply_query_key_layer_scaling: bool = True
fp16_lm_cross_entropy: bool = False
fp32_residual_connection: bool = False
attention_softmax_in_fp32: bool = False

# =========================================================
# dataset
# =========================================================

tokenizer_type: str = "GPT2BPETokenizer"
num_workers: int = 2
mmap_warmup: bool = False
padded_vocab_size: int = 0
make_vocab_size_divisible_by: int = 128
max_position_embeddings: int = 1024

reset_position_ids: bool = False
reset_attention_mask: bool = False
eod_mask_loss: bool = False

# =========================================================
# distributed parallel
# =========================================================

dist_backend: str = None
DDP_impl: str = "native"
gradient_accumulation_fusion: bool = False
use_contiguous_buffers_in_local_ddp: bool = False
