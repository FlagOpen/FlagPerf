from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0

    val_losses: float = 0.0
    val_acc: float = 0.0

    epoch: int = 1
    end_training: bool = False
    converged: bool = False

    init_time = 0
    raw_train_time = 0
    throughoutputs = 0

    def status(self):
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        self.end_training = True
        self.converged = True
