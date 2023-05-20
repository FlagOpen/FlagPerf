# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

from .event import Event, EventManager
from .perf_logger import PerfLogger, LogLevel

STACKLEVEL = 4


class LogEventManager(EventManager):
    """LogEventManager"""

    def __init__(self,
                 local_rank,
                 logger: PerfLogger = None,
                 log_freq: int = 0):
        super(LogEventManager, self).__init__()
        self.log_freq = log_freq
        level = LogLevel.INFO if log_freq > 0 else LogLevel.SUBMITTION
        self.logger = logger or PerfLogger.get_default_logger(rank=local_rank,
                                                              level=level)

    def on_launch_training(self):
        """on launch_training"""
        self._log_event(Event.LAUNCH_TRAINING,
                        "Launch training",
                        stacklevel=STACKLEVEL)

    def on_init_evaluation(self, result: dict):
        """on init_evaluation"""
        self._log_event(Event.INIT_EVALUATION, message=result)

    def on_evaluate(self, result: dict):
        """evaluate event"""
        self._log_event(Event.EVALUATE, message=result)

    def on_backward(self, step: int, loss, optimizer, grad_scaler=None):
        """on backward"""
        pass
        #self._log_event(Event.BACKWARD, f"step: {step}  loss: {loss}")

    def on_init_start(self):
        """on init_start"""
        self._log_event(Event.INIT_START)

    def on_init_end(self):
        """on init_end"""
        self._log_event(Event.INIT_END, message="Finish initialization")

    def on_train_start(self):
        """on train_start"""
        self._log_event(Event.TRAIN_START)

    def on_train_end(self):
        """on train_end"""
        self._log_event(Event.TRAIN_END)

    def on_epoch_begin(self, epoch: int):
        """on epoch_begin"""
        epoch_info = dict(epoch=epoch)
        self._log_event(Event.EPOCH_BEGIN, epoch=epoch_info)

    def on_epoch_end(self, epoch: int, message=None):
        """on epoch_end"""
        epoch_info = dict(epoch=epoch)
        self._log_event(Event.EPOCH_END, epoch=epoch_info, message=message)

    def on_step_begin(self, step: int = None):
        """on step_begin"""
        if (self.log_freq <= 0 or step % self.log_freq != 0) and step != 1:
            return
        self._log_event(Event.STEP_BEGIN, step=step)

    def on_step_end(self, step: int = None, loss=None, message=None):
        """step_end event"""
        if (self.log_freq <= 0 or step % self.log_freq != 0) and step != 1:
            return
        self._log_event(Event.STEP_END, message=message)

    def _log_event(self, event, *args, **kwargs):
        self.logger.log(event, stacklevel=STACKLEVEL, *args, **kwargs)
