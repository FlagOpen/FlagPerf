from torch.optim import AdamW

from .lamb import Lamb


def create_optimizer(name: str, params, config):
    name = name.lower()

    if name == "lamb":
        return Lamb(
            params, lr=config.learning_rate,
            betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_1), eps=1e-6,
            weight_decay=config.weight_decay_rate, adam=False
        )

    if name == "adamw":
        return AdamW(
            params, lr=config.learning_rate,
            betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_2),
            weight_decay=config.weight_decay_rate
        )

    raise RuntimeError(f"Not found optimier {name}.")
