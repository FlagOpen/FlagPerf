# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
    eval_mAP: float = 0.0

    epoch: int = 1
    num_trained_samples = 0
    end_training: bool = False
    converged: bool = False

    traintime = 0.0
    noevaltime = 0.0
    purecomputetime = 0.0

    def status(self):
        """get status"""
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        """converged success"""
        self.end_training = True
        self.converged = True
