
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
model_drop_rate = 0.0
# Drop path rate
model_drop_path_rate = 0.2
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