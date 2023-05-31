from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    """TrainingState dataclass"""
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0

    loss: float = 0.0
    eval_map_bbox :float = 0.0
    eval_map_segm :float = 0.0


    epoch: int = 1
    num_trained_samples = 0
    end_training: bool = False
    converged: bool = False

    init_time = 0
    raw_train_time = 0

    train_start_timestamp = 0

    def status(self):
        """get status"""
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        """converged success"""
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
        """convert kwargs to dict"""
        state_dict = dict()

        for var_name, value in self.__dict__.items():
            if not var_name.startswith("_") and self._is_property(value):
                state_dict[var_name] = value

        lr = self._trainer.lr_scheduler.get_lr()
        if isinstance(lr, (tuple, list)):
            lr = lr[0]

        state_dict["learning_rate"] = lr
        exclude = [
            "eval_loss", "eval_mlm_accuracy", "skipped_steps", "converged",
            "init_time", "raw_train_time"
        ]
        for exkey in exclude:
            if exkey in state_dict:
                state_dict.pop(exkey)

        state_dict.update(kwargs)

        for k, val in state_dict.items():
            if torch.is_tensor(val):
                state_dict[k] = val.item()
        return state_dict
