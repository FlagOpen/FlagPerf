import config


from .linear_warmup_poly_scheduler import LinearWarmupPolyDecayScheduler
from .linear_warmup_scheduler import LinearWarmUpScheduler


def create_scheduler(optimizer, scheduler="poly"):
    if config.warmup_proportion == 0:
        warmup_steps = config.warmup_steps
        warmup_start = config.start_warmup_step
    else:
        warmup_steps = int(config.max_steps * config.warmup_proportion)
        warmup_start = 0

    if scheduler == "linear":
        return LinearWarmUpScheduler(optimizer, warmup_steps, config.max_steps)

    if scheduler == "poly":
        return LinearWarmupPolyDecayScheduler(optimizer, start_warmup_steps=warmup_start,
                                              warmup_steps=warmup_steps,
                                              total_steps=config.max_steps, end_learning_rate=0.0, degree=1.0)

    raise ValueError(f"Not found scheduler {scheduler}.")