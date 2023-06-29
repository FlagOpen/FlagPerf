import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def create_scheduler(config, optimizer, n_iter_per_epoch):
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_warmup_lr = config.train_warmup_lr * config.train_batch_size * config.n_device / 512.0
    linear_scaled_min_lr = config.train_min_lr * config.train_batch_size * config.n_device / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.train_accumulation_steps > 1:
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.train_accumulation_steps
        linear_scaled_min_lr = linear_scaled_min_lr * config.train_accumulation_steps

    config.train_warmup_lr = linear_scaled_warmup_lr
    config.train_min_lr = linear_scaled_min_lr

    num_steps = int(config.train_epochs * n_iter_per_epoch)
    warmup_steps = int(config.train_warmup_epochs * n_iter_per_epoch)
    decay_steps = int(config.train_lr_scheduler_decay_epochs * n_iter_per_epoch)

    lr_scheduler = None
    if config.train_lr_scheduler_name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config.train_min_lr,
            warmup_lr_init=config.train_warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.train_lr_scheduler_name == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.train_warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.train_lr_scheduler_name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.train_lr_scheduler_decay_rate,
            warmup_lr_init=config.train_warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

