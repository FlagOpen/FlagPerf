from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    """TrainingState dataclass"""
    _trainer = None

    val_loss: float = 0.0

    epoch: int = 1
    end_training: bool = False
    converged: bool = False

    init_time = 0
    raw_train_time = 0

    train_start_timestamp = 0

    def converged_success(self):
        """converged success"""
        self.end_training = True
        self.converged = True
