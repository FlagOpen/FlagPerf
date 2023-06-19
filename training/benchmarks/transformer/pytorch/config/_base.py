from .model_params import *
# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = "nvidia"
# model name
name: str = "Transformer"
data_dir: str = "/home/datasets_ckpt/transformer/train/data/wmt14_en_de_joined_dict/"

do_train = True
fp16 = False
# =========================================================
# data
# =========================================================

init_checkpoint: str = ""
data = None
restore_file = "checkpoint_last.pt"
train_subset = "train"
valid_subset = "valid"
gen_subset = "test"

# =========================================================
# Model
# =========================================================
arch = "transformer_wmt_en_de_big_t2t"
lr_scheduler = "inverse_sqrt"
seed = 1


# =========================================================
# loss scale
# =========================================================


# =========================================================
# train && evaluate
# =========================================================
log_freq: int = 50

gradient_accumulation_steps = 1
dist_backend: str = 'nccl'
device: str = None
n_device: int = 1

adam_betas = [0.9, 0.997]
adam_eps = 1e-09
amp = False
cpu = False
criterion = "label_smoothed_cross_entropy"
distributed_rank = 0
distributed_world_size = 1
file = None
lenpen = 1
local_rank = 0
lr = [0.000846]
lr_shrink = 0.1
max_epoch = 2
max_sentences = None
max_sentences_valid = None
max_tokens = 5120
max_update = None
min_lr = 0.0
momentum = 0.99
num_shards = 1
online_eval = True
optimizer = "adam"
path = None
sampling = False
sampling_temperature = 1
sampling_topk = -1
save_dir = "results"
save_interval = 1
save_predictions = False
shard_id = 0
share_all_embeddings = True
share_decoder_input_output_embed = False
source_lang = None
target_bleu = 27.0
target_loss = 3.9
target_lang = None
test_cased_bleu = False
unkpen = 0
unnormalized = False
update_freq = [1]
validate_interval = 1
warmup_init_lr = 0.0
warmup_updates = 4000
weight_decay = 0.0

epochs: int = max_epoch
