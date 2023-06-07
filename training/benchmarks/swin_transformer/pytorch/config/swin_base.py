
# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = None
# model name
name: str = "swin_transformer"

do_train = True
fp16 = True

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 10

# Stop training after reaching this accuracy
target_acc1: float = 70.634
# Whether to read local rank from ENVVAR
use_env: bool = True
# device
device: str = None
n_device: int = 1
dist_backend: str = 'nccl'
gradient_accumulation_steps = 1
data_dir = ""
train_data = ""
eval_data = ""

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
# Batch size for a single GPU, could be overwritten by command line argument
train_batch_size = 64
# Path to dataset, could be overwritten by command line argument
train_data_path = ''
# Dataset name
data_dataset = 'imagenet'
# Input image size
data_img_size = 224
# Interpolation to resize image (random, bilinear, bicubic)
data_interpolation = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
data_zip_mode = False
# Cache Data in Memory, could be overwritten by command line argument
data_cache_mode = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
data_pin_memory = True
# Number of data loading threads
data_num_workers = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# Model type
model_type = 'swin'
# Model name
model_name = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
model_pretrained = ''
# Checkpoint to resume, could be overwritten by command line argument
model_resume = ''
# Number of classes, overwritten in data preparation
model_num_classes = 1000
# Dropout rate
model_drop_rate = 0.2
# Drop path rate
model_drop_path_rate = 0.1
# Label Smoothing
model_label_smoothing = 0.1

# Swin Transformer parameters
model_swin_patch_size = 4
model_swin_in_chans = 3
model_swin_embed_dim = 96
model_swin_depths = [2, 2, 6, 2]
model_swin_num_heads = [3, 6, 12, 24]
model_swin_window_size = 7
model_swin_mlp_ratio = 4.
model_swin_qkv_bias = True
model_swin_qk_scale = None
model_swin_ape = False
model_swin_patch_norm = True

# Swin Transformer V2 parameters
model_swinv2_patch_size = 4
model_swinv2_in_chans = 3
model_swinv2_embed_dim = 96
model_swinv2_depths = [2, 2, 6, 2]
model_swinv2_num_heads = [3, 6, 12, 24]
model_swinv2_window_size = 7
model_swinv2_mlp_ratio = 4.
model_swinv2_qkv_bias = True
model_swinv2_ape = False
model_swinv2_patch_norm = True
model_swinv2_pretrained_window_sizes = [0, 0, 0, 0]

# Swin Transformer MoE parameters
model_swin_moe_patch_size = 4
model_swin_moe_in_chans = 3
model_swin_moe_embed_dim = 96
model_swin_moe_depths = [2, 2, 6, 2]
model_swin_moe_num_heads = [3, 6, 12, 24]
model_swin_moe_window_size = 7
model_swin_moe_mlp_ratio = 4.
model_swin_moe_qkv_bias = True
model_swin_moe_qk_scale = None
model_swin_moe_ape = False
model_swin_moe_patch_norm = True
model_swin_moe_mlp_fc2_bias = True
model_swin_moe_init_std = 0.02
model_swin_moe_pretrained_window_sizes = [0, 0, 0, 0]
model_swin_moe_moe_blocks = [[-1], [-1], [-1], [-1]]
model_swin_moe_num_local_experts = 1
model_swin_moe_top_value = 1
model_swin_moe_capacity_factor = 1.25
model_swin_moe_cosine_router = False
model_swin_moe_normalize_gate = False
model_swin_moe_use_bpr = True
model_swin_moe_is_gshard_loss = False
model_swin_moe_gate_noise = 1.0
model_swin_moe_cosine_router_dim = 256
model_swin_moe_cosine_router_init_t = 0.5
model_swin_moe_moe_drop = 0.0
model_swin_moe_aux_loss_weight = 0.01

# Swin MLP parameters
model_swin_mlp_patch_size = 4
model_swin_mlp_in_chans = 3
model_swin_mlp_embed_dim = 96
model_swin_mlp_depths = [2, 2, 6, 2]
model_swin_mlp_num_heads = [3, 6, 12, 24]
model_swin_mlp_window_size = 7
model_swin_mlp_mlp_ratio = 4.
model_swin_mlp_ape = False
model_swin_mlp_patch_norm = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
train_start_epoch = 0
train_epochs = 300
train_warmup_epochs = 20
train_weight_decay = 0.05
train_base_lr = 5e-4
train_warmup_lr = 5e-7
train_min_lr = 5e-6
# Clip gradient norm
train_clip_grad = 5.0
# Auto resume from latest checkpoint
train_auto_resume = True
# Gradient accumulation steps
# could be overwritten by command line argument
train_accumulation_steps = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
train_use_checkpoint = False

# LR scheduler
train_lr_scheduler_name = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
train_lr_scheduler_decay_epochs = 30
# LR decay rate, used in StepLRScheduler
train_lr_scheduler_decay_rate = 0.1

# Optimizer
train_optimizer_name = 'adamw'
# Optimizer Epsilon
train_optimizer_eps = 1e-8
# Optimizer Betas
train_optimizer_betas = (0.9, 0.999)
# SGD momentum
train_optimizer_momentum = 0.9

# MoE
# Only save model on master device
train_moe_save_master = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
# Color jitter factor
aug_color_jitter = 0.4
# Use AutoAugment policy. "v0" or "original"
aug_auto_augment = 'rand-m9-mstd0.5-inc1'
# Random erase prob
aug_reprob = 0.25
# Random erase mode
aug_remode = 'pixel'
# Random erase count
aug_recount = 1
# Mixup alpha, mixup enabled if > 0
aug_mixup = 0.8
# Cutmix alpha, cutmix enabled if > 0
aug_cutmix = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
aug_cutmix_minmax = None
# Probability of performing mixup or cutmix when either/both is enabled
aug_mixup_prob = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
aug_mixup_switch_prob = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
aug_mixup_mode = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
# Whether to use center crop when testing
test_crop = True
# Whether to use SequentialSampler as validation sampler
test_sequential = False
test_shuffle = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Enable Pytorch automatic mixed precision (amp).
amp_enable = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
amp_opt_level = ''
# Path to output folder, overwritten by command line argument
output = ''
# Tag of experiment, overwritten by command line argument
tag = 'default'
# Frequency to save checkpoint
save_freq = 1
# Frequency to logging info
print_freq = 10
# Fixed random seed
seed = 0
# Perform evaluation only, overwritten by command line argument
eval_mode = False
# Test throughput only, overwritten by command line argument
throughput_mode = False
# local rank for DistributedDataParallel, given by command line argument
local_rank = 0
# for acceleration
fused_window_process = False

