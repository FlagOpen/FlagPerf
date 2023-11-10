from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    """TrainingState dataclass"""
    global_steps = 0

    loss: float = 0.0
    eval_mAP: float = 0.0

    epoch: int = 1
    end_training: bool = False
    converged: bool = False

    num_trained_samples: int = 0

    init_time = 0
    raw_train_time = 0
    no_eval_time = 0.0
    pure_compute_time = 0.0

    def converged_success(self):
        """converged success"""
        self.end_training = True
        self.converged = True
