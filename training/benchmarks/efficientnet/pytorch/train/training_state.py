from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0
    skipped_steps = 0

    loss: float = 0.0
    acc1: float = 0.0
    acc5: float = 0.0

    eval_loss: float = 0.0
    eval_acc1: float = 0.0
    eval_acc5: float = 0.0

    epoch: int = 0
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

        lr = self._trainer.lr_scheduler.get_last_lr()
        if isinstance(lr, (tuple, list)):
            lr = lr[0]
        state_dict["learning_rate"] = lr
        exclude = [
            "eval_loss", "eval_acc1", "eval_acc5", "skipped_steps",
            "converged", "init_time", "raw_train_time"
        ]
        for exkey in exclude:
            if exkey in state_dict:
                state_dict.pop(exkey)

        state_dict.update(kwargs)

        for k in state_dict.keys():
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].item()

        return state_dict

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != '_trainer'
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
