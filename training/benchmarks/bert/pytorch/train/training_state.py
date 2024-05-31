from dataclasses import dataclass

import torch
import utils


@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0
    skipped_steps = 0
    iter_dataloader_idx = 0

    loss: float = 0.0
    mlm_acc: float = 0.0

    epoch: int = 1
    num_trained_samples = 0
    end_training: bool = False
    converged: bool = False

    eval_loss = 0
    eval_mlm_accuracy = 0

    init_time = 0
    raw_train_time = 0

    no_eval_time = 0.0
    pure_compute_time = 0.0

    def status(self):
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        self.end_training = True
        self.converged = True

    def to_dict(self, **kwargs):
        state_dict = dict()

        for var_name, value in self.__dict__.items():
            if not var_name.startswith("_") and utils.is_property(value):
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

        for k in state_dict.keys():
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].item()

        return state_dict
