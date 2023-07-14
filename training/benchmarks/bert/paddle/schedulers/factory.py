import config

from .linear_warmup_poly_scheduler import LinearWarmupPolyDecayScheduler
from .linear_warmup_scheduler import LinearWarmUpScheduler
from paddlenlp.transformers import PolyDecayWithWarmup, CosineDecayWithWarmup


def create_scheduler(optimizer, scheduler="poly"):
    if config.warmup_steps == 0:
        warmup_steps = int(config.max_steps * config.warmup_proportion)
        warmup_start = 0

    else:
        warmup_steps = config.warmup_steps
        warmup_start = config.start_warmup_step
    if scheduler == "linear":
        return LinearWarmUpScheduler(optimizer, warmup_steps, config.max_steps)

    if scheduler == "poly":
        return LinearWarmupPolyDecayScheduler(
                    startup_warmup_steps=warmup_start,
                    warmup_steps=warmup_steps,
                    total_steps=config.max_steps,
                    base_lr=config.learning_rate,
                    end_lr=0.0,
                    degree=1.0)

        # return PolyDecayWithWarmup(learning_rate=config.learning_rate,
        #                            warmup=warmup_steps,
        #                            total_steps=config.max_steps,
        #                            lr_end=0.0,
        #                            power=1.0)
    if scheduler == "cos":
        print("++++++++++++++++++++++cos+++++++++++++++++++++")
        return CosineDecayWithWarmup(config.learning_rate,
                                     config.max_steps,
                                     warmup_steps)
    raise ValueError(f"Not found scheduler {scheduler}.")
