# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from dataclasses import dataclass
import inspect

import torch


@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0

    loss: float = 0.0
    eval_auc: float = 0.0
    best_auc: float = 0.0
    best_epoch: int = 0
    best_validation_loss: float = 1e6

    epoch: int = 1
    num_trained_samples = 0

    # state related
    end_training: bool = False
    converged: bool = False

    # time related
    train_time_start_ts = 0

    init_time = 0
    raw_train_time = 0
    train_time = 0
    no_eval_time = 0
    pure_compute_time = 0.0

    train_start_timestamp = 0

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
