from typing import ClassVar
#from train.event.base import BaseTrainingEventInterface

# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
name: str = "vit_base_patch16_224"
# =========================================================
# data
# =========================================================
log_freq: int = 1
n_device: int = 1




data_dir: str = "/home/data/imagenet"


init_checkpoint: str = ""

# ======================= todo 这些参数要不要

aa: str  =  None
amp  =  False
amp_dtype: str =  "float16"
amp_impl: str  =  "native"
aot_autograd  =  False
aug_repeats  =  0
aug_splits  =  0
batch_size  =  128
bce_loss  =  False
bce_target_thresh: float =  None
bn_eps: float  =  None
bn_momentum: float  =  None
channels_last  =  False
checkpoint_hist  =  10
class_map  =  ""
clip_grad: float =  None
clip_mode  =  "norm"
color_jitter  =  0.4
cooldown_epochs  =  0
crop_pct: float =  None
cutmix  =  0.0
cutmix_minmax: float =  None
data: str =  None
# data_dir  =  "/home/data/train"
dataset  =  ""
dataset_download  =  False
decay_epochs: float =  90
decay_milestones  =  [90, 180, 270]
decay_rate  =  0.1
device  =  "xla"
dist_bn  =  "reduce"
drop: float =  0.0
drop_block: float =  None
drop_connect: float =  None
drop_path: float =  None
epoch_repeats  =  0.0
epochs  =  1
eval_metric  =  "top1"
experiment  =  ""
fast_norm  =  False
fuser  =  ""
gp: str =  None
grad_checkpointing  =  False
hflip  =  0.5
img_size:int  =  None
in_chans  =  None
initial_checkpoint  =  ""
input_size  =  None
interpolation  =  ""
jsd_loss  =  False
layer_decay  =  None
local_rank  =  0
log_interval  =  1
log_wandb  =  False
lr  =  0.4
lr_base  =  0.1
lr_base_scale  =  ""
lr_base_size  =  256
lr_cycle_decay  =  0.5
lr_cycle_limit  =  1
lr_cycle_mul  =  1.0
lr_k_decay  =  1.0
lr_noise  =  None
lr_noise_pct  =  0.67
lr_noise_std  =  1.0
mean  =  None
min_lr  =  0
mixup  =  0.0
mixup_mode  =  "batch"
mixup_off_epoch  =  0
mixup_prob  =  1.0
mixup_switch_prob  =  0.5
model  =  "vit_base_patch16_224"
model_ema  =  False
model_ema_decay  =  0.9998
model_ema_force_cpu  =  False
model_kwargs  =  {}
momentum  =  0.9
no_aug  =  False
no_ddp_bb  =  False
no_prefetcher  =  False
no_resume_opt  =  False
num_classes  =  None
opt  =  "sgd"
opt_betas  =  None
opt_eps  =  None
opt_kwargs  =  {}
output  =  ""
patience_epochs  =  10
pin_mem  =  False
pretrained  =  False
ratio  =  [0.75, 1.3333333333333333]
recount  =  1
recovery_interval  =  0
remode  =  "pixel"
reprob  =  0.0
resplit  =  False
resume  =  ""
save_images  =  False
scale  =  [0.08, 1.0]
sched  =  "cosine"
sched_on_updates  =  False
seed  =  42
smoothing  =  0.1
split_bn  =  False
start_epoch  =  None
std  =  None
sync_bn  =  False
torchcompile  =  None
torchscript  =  False
train_interpolation  =  "random"
train_split  =  "train"
tta  =  0
use_multi_epochs_loader  =  False
val_split  =  "val"
validation_batch_size  =  None
vflip  =  0.0
warmup_epochs  =  5
warmup_lr  =  1e-05
warmup_prefix  =  False
weight_decay  =  2e-05
worker_seeding  =  "all"
workers  =  4

distributed = False  # todo wzd