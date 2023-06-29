from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0

    loss: float = 0.0
    acc1: float = 0.0
    acc5: float = 0.0
    batch_time: float = 0.0
    max_accuracy: float = 0.0
    
    eval_loss: float = 0.0
    eval_acc1: float = 0.0
    eval_acc5: float = 0.0

    epoch: int = 1
    num_trained_samples = 0
    end_training: bool = False
    converged: bool = False

    init_time = 0
    raw_train_time = 0

    def status(self):
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        self.end_training = True
        self.converged = True

    def _is_property(self, value):
        status = [
            not callable(value), not inspect.isclass(value),
            not inspect.ismodule(value), not inspect.ismethod(value),
            not inspect.isfunction(value), not inspect.isbuiltin(value),
            "classmethod object" not in str(value)
        ]
        return all(status)

    def to_dict(self, **kwargs):
        state_dict = dict()

        for var_name, value in self.__dict__.items():
            if not var_name.startswith("_") and self._is_property(value):
                state_dict[var_name] = value

        exclude = [
            "eval_loss", "acc1", "acc5", "max_accuracy", "eval_acc1", "eval_acc5", "skipped_steps",
            "converged", "init_time", "raw_train_time", "batch_time"
        ]
        for exkey in exclude:
            if exkey in state_dict:
                state_dict.pop(exkey)

        state_dict.update(kwargs)

        for k in state_dict.keys():
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].item()

        return state_dict
