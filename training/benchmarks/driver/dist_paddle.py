import os
from contextlib import contextmanager

import paddle
import paddle.distributed as dist
from paddlenlp.trainer import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from paddlenlp.trainer.trainer_utils import IntervalStrategy

from .base import Driver
from .event import Event
from typing import Dict

def barrier():
    if dist.is_initialized():
        dist.barrier()

def is_main_process():
    if dist.is_initialized():
        if "PADDLE_TRAINER_ID" in os.environ:
            return int(os.environ["PADDLE_TRAINER_ID"]) == 0
        else:
            return dist.get_rank() == 0

    return True

class PaddleCallback(TrainerCallback):
    def __init__(self, driver: Driver):
        self.driver = driver

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerState,
        **kwargs
    ):
        self.driver.event(Event.INIT_END)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.TRAIN_START)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.TRAIN_END)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.EPOCH_BEGIN, epoch=state.epoch)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.EPOCH_END, epoch=state.epoch)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.STEP_BEGIN, step=state.global_step + 1)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        logs = kwargs["metrics"]
        logs["global_step"] = state.global_step
        self.driver.event(Event.EVALUATE, result=logs)
        if kwargs["metrics"]["eval_ppl"] < self.driver.config.target_ppl:
            control.should_training_stop = True
            


    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            self.driver.logger.log(Event.STEP_END, message=logs)