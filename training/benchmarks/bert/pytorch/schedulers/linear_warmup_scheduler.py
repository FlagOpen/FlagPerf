import numpy as np

from .base import LRScheduler


class LinearWarmUpScheduler(LRScheduler):

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [
                base_lr * progress / self.warmup for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * max((progress - 1.0) / (self.warmup - 1.0), 0.)
                for base_lr in self.base_lrs
            ]
