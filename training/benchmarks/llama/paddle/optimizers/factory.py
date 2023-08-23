from paddle.optimizer import AdamW
import paddle.nn as nn
import paddle.distributed.fleet as fleet

def create_optimizer(config, model: nn.Layer, lr_scheduler):
    parameter_list = model.parameters()
    decay_parameters = [
        p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
    ]

    def apply_decay_param_fun(x):
        return x in decay_parameters
    
    
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