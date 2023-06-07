# --------------------------------------------------------

# Swin Transformer

# Copyright (c) 2021 Microsoft

# Licensed under The MIT License [see LICENSE for details]

# Written by Ze Liu

# --------------------------------------------------------


from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP


def create_model(config):
    model_type = config.model_type
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.data_img_size,
                                patch_size=config.model_swin_patch_size,
                                in_chans=config.model_swin_in_chans,
                                num_classes=config.model_num_classes,
                                embed_dim=config.model_swin_embed_dim,
                                depths=config.model_swin_depths,
                                num_heads=config.model_swin_num_heads,
                                window_size=config.model_swin_window_size,
                                mlp_ratio=config.model_swin_mlp_ratio,
                                qkv_bias=config.model_swin_qkv_bias,
                                qk_scale=config.model_swin_qk_scale,
                                drop_rate=config.model_drop_rate,
                                drop_path_rate=config.model_drop_path_rate,
                                ape=config.model_swin_ape,
                                patch_norm=config.model_swin_patch_norm,
                                use_checkpoint=config.train_use_checkpoint,
                                fused_window_process=config.fused_window_process)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.data_img_size,
                                  patch_size=config.model_swinv2_patch_size,
                                  in_chans=config.model_swinv2_in_chans,
                                  num_classes=config.model_num_classes,
                                  embed_dim=config.model_swinv2_embed_dim,
                                  depths=config.model_swinv2_depths,
                                  num_heads=config.model_swinv2_num_heads,
                                  window_size=config.model_swinv2_window_size,
                                  mlp_ratio=config.model_swinv2_mlp_ratio,
                                  qkv_bias=config.model_swinv2_qkv_bias,
                                  drop_rate=config.model_drop_rate,
                                  drop_path_rate=config.model_drop_path_rate,
                                  ape=config.model_swinv2_ape,
                                  patch_norm=config.model_swinv2_patch_norm,
                                  use_checkpoint=config.train_use_checkpoint,
                                  pretrained_window_sizes=config.model_swinv2_pretrained_window_sizes)
    elif model_type == 'swin_moe':
        model = SwinTransformerMoE(img_size=config.data_img_size,
                                   patch_size=config.model_swin_moe_patch_size,
                                   in_chans=config.model_swin_moe_in_chans,
                                   num_classes=config.model_num_classes,
                                   embed_dim=config.model_swin_moe_embed_dim,
                                   depths=config.model_swin_moe_depths,
                                   num_heads=config.model_swin_moe_num_heads,
                                   window_size=config.model_swin_moe_window_size,
                                   mlp_ratio=config.model_swin_moe_mlp_ratio,
                                   qkv_bias=config.model_swin_moe_qkv_bias,
                                   qk_scale=config.model_swin_moe_qk_scale,
                                   drop_rate=config.model_drop_rate,
                                   drop_path_rate=config.model_drop_path_rate,
                                   ape=config.model_swin_moe_ape,
                                   patch_norm=config.model_swin_moe_patch_norm,
                                   mlp_fc2_bias=config.model_swin_moe_mlp_fc2_bias,
                                   init_std=config.model_swin_moe_init_std,
                                   use_checkpoint=config.train_use_checkpoint,
                                   pretrained_window_sizes=config.model_swin_moe_pretrained_window_sizes,
                                   moe_blocks=config.model_swin_moe_moe_blocks,
                                   num_local_experts=config.model_swin_moe_num_local_experts,
                                   top_value=config.model_swin_moe_top_value,
                                   capacity_factor=config.model_swin_moe_capacity_factor,
                                   cosine_router=config.model_swin_moe_cosine_router,
                                   normalize_gate=config.model_swin_moe_normalize_gate,
                                   use_bpr=config.model_swin_moe_use_bpr,
                                   is_gshard_loss=config.model_swin_moe_is_gshard_loss,
                                   gate_noise=config.model_swin_moe_gate_noise,
                                   cosine_router_dim=config.model_swin_moe_cosine_router_dim,
                                   cosine_router_init_t=config.model_swin_moe_cosine_router_init_t,
                                   moe_drop=config.model_swin_moe_moe_drop,
                                   aux_loss_weight=config.model_swin_moe_aux_loss_weight)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.data_img_size,
                        patch_size=config.model_swin_mlp_patch_size,
                        in_chans=config.model_swin_mlp_in_chans,
                        num_classes=config.model_num_classes,
                        embed_dim=config.model_swin_mlp_embed_dim,
                        depths=config.model_swin_mlp_depths,
                        num_heads=config.model_swin_mlp_num_heads,
                        window_size=config.model_swin_mlp_window_size,
                        mlp_ratio=config.model_swin_mlp_mlp_ratio,
                        drop_rate=config.model_drop_rate,
                        drop_path_rate=config.model_drop_path_rate,
                        ape=config.model_swin_mlp_ape,
                        patch_norm=config.model_swin_mlp_patch_norm,
                        use_checkpoint=config.train_use_checkpoint)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
