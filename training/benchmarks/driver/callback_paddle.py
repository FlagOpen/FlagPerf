from paddlenlp.trainer import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .base import Driver
from .event import Event


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
        self.driver.event(Event.STEP_BEGIN, step=state.global_step)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.STEP_END, step=state.global_step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.driver.event(Event.EVALUATE)
