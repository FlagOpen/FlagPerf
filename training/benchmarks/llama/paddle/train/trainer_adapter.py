import os
import sys
import paddle
import paddle.distributed as dist
import paddle.amp.auto_cast as autocast
import paddle.distributed.fleet as fleet
from paddle.optimizer import Optimizer
from paddle import nn, Tensor
from typing import Tuple

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import dist_paddle


def convert_model(config, model: nn.Layer) -> nn.Layer:
    return model


def model_to_fp16(config, model: nn.Layer, optimizer):
    paddle.amp.decorate(models=model, level=config.fp16_opt_level)
    decorated = paddle.amp.decorate(models=model, optimizers=optimizer, level=config.fp16_opt_level)
    model, optimizer = decorated

    return model, optimizer

def autocast_smart_context_manager(config):
    """
    A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
    arguments, depending on the situation.
    """
    amp_dtype = "float16" if config.fp16 else "bfloat16"

    if config.fp16:
        custom_black_list = ["reduce_sum", "c_softmax_with_cross_entropy"]
        custom_white_list = []
        if config.fp16_opt_level == "O2":
            custom_white_list.extend(["lookup_table", "lookup_table_v2"])

        if config.bf16 and config.fp16_opt_level == "O2":
            custom_black_list.append("c_embedding")

        if config.amp_custom_white_list is not None:
            custom_white_list.extend(config.amp_custom_white_list)
        if config.amp_custom_black_list is not None:
            custom_black_list.extend(config.amp_custom_black_list)

        ctx_manager = autocast(
            True,
            custom_black_list=set(custom_black_list),
            custom_white_list=set(custom_white_list),
            level=config.fp16_opt_level,
            dtype=amp_dtype,
        )

    return ctx_manager


def model_to_ddp(config, model: nn.Layer) -> nn.Layer:
    if dist_paddle.get_world_size() > 1:
        model = paddle.DataParallel(model)
    # Recompute
    if config.use_recompute:
        def fn(layer):
            if hasattr(layer, "enable_recompute") and (
                layer.enable_recompute is False or layer.enable_recompute == 0
            ):
                layer.enable_recompute = True

        model.apply(fn)
    return model

def create_grad_scaler(config):
    scaler = paddle.amp.GradScaler(init_loss_scaling=config.scale_loss)
    if config.sharding:
        from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
            GroupShardedScaler,
        )
        scaler = GroupShardedScaler(scaler)
    return scaler

def train_on_sharding(config, model, optimizer, grad_scaler):
    sharding_stage_map = {"stage1":"os", "stage2":"os_g", "stage3":"p_g_os"}
    model, optimizer, grad_scaler = dist_paddle.group_sharded_parallel(model, optimizer, 
                                    sharding_stage_map[config.sharding], scaler=grad_scaler)
    return model, optimizer, grad_scaler


# def backward(config, step: int, loss: Tensor, optimizer, lr_scheduler, 
#              do_grad_scaling, scaler, model, **kwarg):
#     if do_grad_scaling:
#         scaler.scale(loss).backward()
#     else:
#         loss.backward()

#     if step % config.gradient_accumulation_steps == 0:
#         if do_grad_scaling:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer_was_run = not scaler._cache_founf_inf
#         else:
#             optimizer.step()
#         if optimizer_was_run:
#             lr_scheduler.step()
#         # for n, p in model.named_parameters():
#         #     ic(n, p, p.grad)
#         optimizer.clear_grad()

        
def backward(config, step: int, loss: Tensor, optimizer, lr_scheduler, 
             do_grad_scaling, scaler, **kwarg):
    if do_grad_scaling:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if step % config.gradient_accumulation_steps == 0:
        if do_grad_scaling:
            scaler.step(optimizer)
            scaler.update()
            optimizer_was_run = not scaler._cache_founf_inf
        else:
            optimizer.step()

        if optimizer_was_run:
            lr_scheduler.step()

        optimizer.clear_grad()