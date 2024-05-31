import torch
import torch.distributed as dist

from torch.optim import Optimizer
from torch import nn, Tensor
from typing import Tuple

from model.models import gpt2_get_params_for_weight_decay_optimization
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from model.fp16 import FP16_Module
from model.fp16 import FP16_Optimizer


def convert_model(config, model: nn.Module) -> nn.Module:
    state_dict = model.state_dict()
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    for i in range(config.num_layers):
        model.transformer.layers[i].input_layernorm = LayerNorm(
            config.hidden_size, config.layernorm_epsilon)
        model.transformer.layers[i].post_attention_layernorm = LayerNorm(
            config.hidden_size, config.layernorm_epsilon)
    model.transformer.final_layernorm = LayerNorm(config.hidden_size,
                                                  config.layernorm_epsilon)

    model.load_state_dict(state_dict, strict=True)
    return model


def create_optimizer(config, model: nn.Module) -> Optimizer:
    param_groups = gpt2_get_params_for_weight_decay_optimization(model)
    from apex.optimizers import FusedAdam as Adam
    optimizer = Adam(param_groups,
                     lr=config.learning_rate,
                     weight_decay=config.weight_decay_rate)

    return optimizer


def model_to_fp16(config, model: nn.Module,
                  optimizer: Optimizer) -> Tuple[nn.Module, Optimizer]:
    args = config
    if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                static_loss_scale=args.loss_scale,
                                dynamic_loss_scale=args.dynamic_loss_scale,
                                dynamic_loss_args={
                                    'scale_window': args.loss_scale_window,
                                    'min_scale': args.min_scale,
                                    'delayed_shift': args.hysteresis
                                })

    return model, optimizer


def model_to_ddp(config, model: nn.Module) -> nn.Module:
    use_ddp = dist.is_initialized()

    if use_ddp:
        if config.ddp_type == 'native':
            model = NativeDDP(
                model,
                device_ids=[config.local_rank],
                bucket_cap_mb=100,
                gradient_as_bucket_view=config.use_gradient_as_bucket_view)
        elif config.ddp_type == 'apex':
            from apex.parallel import DistributedDataParallel as APEX_DDP
            model = APEX_DDP(
                model,
                message_size=250000000,
                delay_allreduce=True,
                gradient_predivide_factor=torch.distributed.get_world_size())
        else:
            assert False, "Invalid DDP type"
    return model


def create_grad_scaler():
    return None


def backward(config, step: int, loss: Tensor, optimizer, **kwarg):
    if config.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()

    if step % config.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
