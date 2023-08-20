from paddle.optimizer import AdamW
import paddle.nn as nn


def create_optimizer(name: str, params, config, lr_scheduler):
    if name == "adamw":
        return AdamW(parameters=params,
                     learning_rate=lr_scheduler,
                     beta1=config.adam_beta1,
                     beta2=config.adam_beta2,
                     epsilon=config.adam_epsilon,
                     weight_decay=config.weight_decay,
                     grad_clip=nn.ClipGradByGlobalNorm(config.max_grad_norm)
                     if config.max_grad_norm > 0
                     else None)

    raise RuntimeError(f"Not found optimizer {name}.")