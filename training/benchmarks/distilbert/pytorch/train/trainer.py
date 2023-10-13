# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import os
import sys
import math

import torch
import torch.utils.data
from torch.types import Device

from model import create_model
from optimizers import create_optimizer
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


class Trainer:
    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

    def init(self, train_dataloader):
        self.model, self.model_config, self.tokenizer = create_model(
            self.config)
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_ddp(self.config, self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, train_dataloader,
                                             self.config)


    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch


    def train_one_epoch(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        no_eval_start = time.time()
        for _, data in enumerate(dataloader):
            data = self.process_batch(data, self.device)

            pure_compute_start = time.time()
            state.global_steps += 1
            state.num_trained_samples = state.global_steps * dist_pytorch.global_batch_size(
                self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(data)

            train_end = time.time()
            state.pure_compute_time += train_end - pure_compute_start
            state.no_eval_time += train_end - no_eval_start

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                sequences_per_second = state.num_trained_samples / state.no_eval_time
                other_state["seq/s"] = sequences_per_second

            step_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END,
                         message=step_info,
                         step=state.global_steps,
                         loss=state.loss)
            
            no_eval_start = time.time()

        driver.event(Event.EPOCH_END, state.epoch)
        eval_start = time.time()
        state.acc = self.evaluator.evaluate(self)
        eval_result = dict(
            global_steps=state.global_steps,
            acc=state.acc,
            time=time.time() - eval_start)
        driver.event(Event.EVALUATE, eval_result)
        self.detect_training_status(state)
        

    def train_one_step(self, data):

        state = self.training_state
        self.model.train()

        outputs = self.model(**data)
        #loss 为标量
        loss = outputs["loss"].item()
        state.loss = loss
        self.adapter.backward(self.config, state.global_steps, outputs["loss"],
                              self.optimizer)
        self.lr_scheduler.step()
        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer)


    def detect_training_status(self, state: TrainingState):
        if state.acc >= self.config.target_acc:
            state.converged_success()
            state.end_training = True

        if state.epoch >= self.config.max_epoch:
            state.end_training = True

        return state.end_training

