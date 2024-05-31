from transformers import get_scheduler


def create_scheduler(optimizer, max_steps):
    lr_scheduler = get_scheduler(name="linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=max_steps)
    return lr_scheduler
