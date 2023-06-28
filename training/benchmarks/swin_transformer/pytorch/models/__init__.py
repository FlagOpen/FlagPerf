from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2


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
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
