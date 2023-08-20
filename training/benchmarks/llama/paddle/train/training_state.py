from dataclasses import dataclass

import paddle
import inspect
from icecream import ic

@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0
    skipped_steps = 0
    iter_dataloader_idx = 0

    loss: float = 0.0
    epoch: int = 1
    num_trained_samples = 0
    end_training: bool = False
    converged: bool = False

    eval_avg_loss = 0
    eval_embedding_average = 0

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

        lr = self._trainer.optimizer.get_lr()

        if isinstance(lr, (tuple, list)):
            lr = lr[0]

        state_dict["learning_rate"] = "{:.3e}".format(lr)
        state_dict["loss"] = self.loss.detach()

        exclude = [
            "eval_avg_loss", "eval_embedding_average", "skipped_steps",
            "converged", "init_time", "raw_train_time"
        ]
        for exkey in exclude:
            if exkey in state_dict:
                state_dict.pop(exkey)

        state_dict.update(kwargs)

        for k in state_dict.keys():
            if paddle.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].item()

        return state_dict