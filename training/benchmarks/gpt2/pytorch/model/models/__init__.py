# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

try:
    from ..layers.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
except Exception:
    # Make LayerNorm has the same parameters as FusedLayerNorm
    from torch.nn import LayerNorm as TorchLayerNorm
    class LayerNorm(TorchLayerNorm):
        """Inherit from torch.nn.LayerNorm but eliminate extra kwargs"""
        def __init__(self, normalized_shape, eps=1e-5,
                    no_persist_layer_norm=True,
                    sequence_parallel=False,
                    apply_layernorm_1p=False):
                super().__init__(
                    normalized_shape, eps = eps)

from .utils import RMSNorm
from .gpt_model import GPTModel
from .language_model import get_language_model
from .module import Float16Module
