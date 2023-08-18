import paddle
import paddle.distributed as dist

from paddle.optimizer import Optimizer
import paddle.amp.auto_cast as autocast
from paddle import nn, Tensor
from typing import Tuple
from icecream import ic

def convert_model(config, model: nn.Layer) -> nn.Layer:
    return model


def create_optimizer(config, model: nn.Layer, lr_scheduler) -> Optimizer:
    parameter_list = model.parameters()
    decay_parameters = [
        p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
    ]

    def apply_decay_param_fun(x):
        return x in decay_parameters
    
    from paddle.optimizer import AdamW
    optimizer = AdamW(parameters=parameter_list,
                     learning_rate=lr_scheduler,
                     apply_decay_param_fun=apply_decay_param_fun,
                     beta1=config.adam_beta1,
                     beta2=config.adam_beta2,
                     epsilon=config.adam_epsilon,
                     weight_decay=config.weight_decay,
                     grad_clip=nn.ClipGradByGlobalNorm(config.max_grad_norm)
                     if config.max_grad_norm > 0
                     else None,
                     multi_precision=True
                    )
    return optimizer


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
            # https://github.com/PaddlePaddle/Paddle/blob/eb97f4f0adca40b16a309b927e480178beb8ae96/python/paddle/amp/amp_lists.py#L85-L86
            # the lookup_table is in black_list, but in O2, we need it return fp16
            custom_white_list.extend(["lookup_table", "lookup_table_v2"])

        if config.bf16 and config.fp16_opt_level == "O2":
            # c_embedding not support bf16 yet
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
    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model)
    return model


def create_grad_scaler(config):
    scaler = paddle.amp.GradScaler(init_loss_scaling=config.scale_loss)
    return scaler


def backward(config, step: int, loss: Tensor, optimizer, lr_scheduler, 
             do_grad_scaling, scaler, model, **kwarg):
    if do_grad_scaling:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if step % config.gradient_accumulation_steps == 0:
        if do_grad_scaling:
            # ic(optimizer.state_dict())
            # ic(self.lr_scheduler.last_epoch, self.lr_scheduler.decay_step, self.lr_scheduler.get_lr())
            # ic(optimizer.get_lr(), lr_scheduler.get_lr())
            scaler.step(optimizer)
            scaler.update()
            optimizer_was_run = not scaler._cache_founf_inf
        else:
            optimizer.step()
        if optimizer_was_run:
            lr_scheduler.step()
        # for n, p in model.named_parameters():
        #     ic(n, p, p.grad)
        optimizer.clear_grad()
    
    return loss.detach()
        
    