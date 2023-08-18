from paddlenlp.transformers import CosineAnnealingWithWarmupDecay, LinearAnnealingWithWarmupDecay
from icecream import ic

def create_scheduler(config):
    if config.decay_steps is None:
        config.decay_steps = config.max_steps
    warmup_steps = config.warmup_ratio * config.max_steps

    lr_scheduler = None
    if config.lr_scheduler_type == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=config.learning_rate,
            min_lr=config.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=config.decay_steps,
            last_epoch=0,
        )
    elif config.lr_scheduler_type== "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=config.learning_rate,
            min_lr=config.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=config.decay_steps,
            last_epoch=0,
        )
    return lr_scheduler