# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import time
import os
import sys

import torch
from torch.types import Device

from model import create_model
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState
from model.losses.cross_entropy import cross_entropy
from train.utils import get_batch

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import Driver, Event, dist_pytorch

def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, 'module'):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_splits, num_attention_heads_per_partition,
             hidden_size_per_attention_head) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_attention_heads_per_partition,
             hidden_size_per_attention_head, num_splits) +\
             input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


class Trainer():

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.config = config
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None

        self.device = device
        self.optimizer = None
        self.bert_config = None
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.overflow_buf = None

    def init(self):
        self.model_config, self.model = create_model(self.config)
        self.model = self._init_model(self.model, self.device)
        self.model = self.adapter.convert_model(self.config, self.model)
        self.model = self.model.to(self.config.device)

        self.optimizer = self.adapter.create_optimizer(self.config, self.model)
        self.model, self.optimizer = self.adapter.model_to_fp16(
            self.config, self.model, self.optimizer)
        self.model = self.adapter.model_to_ddp(self.config, self.model)
        self.lr_scheduler = create_scheduler(self.optimizer)

    def _init_model(self, model, device):
        model = model.to(device)
        return model

    def train_one_epoch(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        no_eval_start = time.time()
        for _, data in enumerate(dataloader):
            data['text'] = data['text'].to(self.device)
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data)

            pure_compute_start = time.time()
            state.global_steps += 1
            state.num_trained_samples = state.global_steps * dist_pytorch.global_batch_size(
                self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(tokens, position_ids, attention_mask, labels, loss_mask)

            train_end = time.time()
            state.pure_compute_time += train_end - pure_compute_start
            state.no_eval_time += train_end - no_eval_start
            
            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                sequences_per_second = state.num_trained_samples / state.no_eval_time
                other_state["seq/s"] = sequences_per_second

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_lambada_acc = self.evaluator.evaluate(
                    self)
                eval_end = time.time()
                eval_result = dict(
                    global_steps=state.global_steps,
                    eval_lambada_acc=state.eval_lambada_acc,
                    time=eval_end - eval_start)

            end_training = self.detect_training_status(state)

            step_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END,
                         message=step_info,
                         step=state.global_steps,
                         loss=state.loss)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if end_training:
                break
            no_eval_start = time.time()

        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, tokens, position_ids, attention_mask, labels, loss_mask):

        state = self.training_state
        self.model.train()

        losses = self.model(tokens, position_ids, attention_mask, labels=labels)
        #loss 为标量
        loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.view(-1).sum()
        state.loss = loss
        self.adapter.backward(state.global_steps, loss,
                              self.optimizer, self.lr_scheduler)
        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer)


    def detect_training_status(self, state: TrainingState):
        if state.eval_lambada_acc >= self.config.target_acc:
            state.converged_success()

        if state.global_steps >= self.config.max_steps:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            self.config.test_data_prefix is not None,
            state.num_trained_samples >= self.config.eval_iter_start_samples,
            self.config.eval_interval_samples > 0,
            state.global_steps > 1,
            state.global_steps %
            math.ceil(self.config.eval_interval_samples /
                      dist_pytorch.global_batch_size(self.config)) == 0,
        ])

        return do_eval or state.global_steps >= self.config.max_steps

