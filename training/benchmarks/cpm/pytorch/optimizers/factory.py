from torch.optim import AdamW


def create_optimizer(name: str, params, config):
    name = name.lower()

    if name == "adamw":
        return AdamW(params,
                     lr=config.learning_rate,
                     betas=(config.beta_1, config.beta_2),
                     weight_decay=config.weight_decay_rate)

    raise RuntimeError(f"Not found optimizer {name}.")
