# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import sys

import numpy as np
import torch
from torch.types import Device

from model import create_model, create_model_config
from model.loss.loss_function import get_loss_function
from model.data.data_function import batch_to_gpu
from optimizers import create_optimizer
from .utils import reduce_tensor
from train.evaluator import Evaluator
from train.training_state import TrainingState
import config

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 world_size, train_dataloader):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.model_config = None
        self.evaluator = evaluator
        self.global_batch_size = None
        self.criterion = None
        self.world_size = world_size
        self.train_dataloader = train_dataloader

    def init(self):
        self.model_config = create_model_config(config)
        self.model = create_model(config)
        self.model = self.adapter.model_to_fp16(self.model, self.config)
        self.model = self.adapter.model_to_ddp(self.model, self.config)
        self.model.train()

        self.criterion = get_loss_function()
        self.optimizer = create_optimizer(self.model, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler(self.config)
        torch.backends.cudnn.enabled = self.config.cudnn_enabled
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark

    def train_one_epoch(self, train_dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        if self.config.distributed:
            self.train_dataloader.sampler.set_epoch(state.epoch)

        no_eval_start_time = time.time()

        for batch in train_dataloader:
            self.train_one_step(batch)
        state.no_eval_time += time.time() - no_eval_start_time

        val_loss, _ = self.evaluator.evaluate(self)
        state.val_loss = val_loss

        epoch_data = {
            "val_loss": val_loss,
            "epoch": state.epoch,
            "global_steps": state.global_steps,
            "num_trained_samples": state.num_mels,
            "timestamp": int(time.time()),
        }
        print(epoch_data)

        driver.event(Event.EPOCH_END, state.epoch)
        self.detect_training_status()

    def train_one_step(self, batch):
        driver = self.driver
        state = self.training_state
        args = self.config
        state.global_steps += 1

        adjust_learning_rate(self.training_state.epoch, self.optimizer,
                             args.learning_rate, args.lr_anneal_steps,
                             args.lr_anneal_factor)

        self.model.zero_grad()
        x, y, len_x = batch_to_gpu(batch)

        pure_compute_start_time = time.time()

        loss = self.adapter.calculate_loss(self.model, self.config,
                                           self.criterion, x, y)

        if args.amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           args.grad_clip_thresh)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           args.grad_clip_thresh)
            self.optimizer.step()
        state.pure_compute_time += time.time() - pure_compute_start_time

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, self.world_size).item()
        else:
            reduced_loss = loss.item()

        if np.isnan(reduced_loss):
            raise Exception("loss is NaN")

        self.model.zero_grad(set_to_none=True)

        state.train_loss = reduced_loss
        state.num_mels += len_x.item() * self.world_size
        step_info = dict(
            step=state.global_steps,
            train_loss=reduced_loss,
            num_trained_samples=state.num_mels,
        )

        print(f"step_info:{step_info}")
        driver.event(Event.STEP_END, state.global_steps)

    def detect_training_status(self):
        config = self.config
        state = self.training_state
        # for loss: the smaller, the better
        if state.val_loss <= config.target_val_loss:
            state.converged_success()

        if state.epoch >= config.max_epochs:
            state.end_training = True

        return state.end_training


def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps,
                         anneal_factor):
    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1**(p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor**p)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
