from torch import nn
from torch import distributed as dist
from torch import optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import config

def convert_model(model: nn.Module) -> nn.Module:
    return model


def create_optimizer(model, config):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.train_base_lr * config.train_batch_size * config.n_device / 512.0

    # gradient accumulation also need to scale the learning rate
    if config.train_accumulation_steps > 1:
        linear_scaled_lr = linear_scaled_lr * config.train_accumulation_steps

    config.train_base_lr = linear_scaled_lr

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.train_optimizer_name.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.train_optimizer_momentum, nesterov=True,
                              lr=config.train_base_lr, weight_decay=config.train_weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.train_optimizer_eps, betas=config.train_optimizer_betas,
                                lr=config.train_base_lr, weight_decay=config.train_weight_decay)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[config.local_rank], broadcast_buffers=False)
    return model


