import math
import time
import os
import sys
import pdb

import paddle
import paddle.nn as nn

import numpy as np
from model import create_model
from schedulers import create_scheduler
from optimizers import create_optimizer

from train.evaluator import Evaluator
from train.training_state import TrainingState

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import Driver, Event, dist_paddle
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import paddle.profiler as profiler

from memory_profiler import profile
 

class Trainer():

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device, config):
        super(Trainer, self).__init__()
        self.config = config
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None

        self.device = device
        self.optimizer = None
        self.llama_config = None
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.global_steps = 0
        self.overflow_buf = None
        self.do_grad_scaling = False
        self.tr_loss = paddle.to_tensor(0.0)
    

    # @profile(precision=4, stream=open("memory_profiler_train_init.log", "w+"))
    def init(self):
        self.model_config, self.model = create_model(self.config)
        # self.model = self._init_model(self.model)

        self.model = self.adapter.convert_model(self.config, self.model)
        self.lr_scheduler = create_scheduler(self.config)
        self.optimizer = create_optimizer(self.config, self.model, self.lr_scheduler)

        # Mixed precision
        if self.config.fp16:
            self.do_grad_scaling = True
            self.model, self.optimizer = self.adapter.model_to_fp16(self.config, self.model, self.optimizer)
            self.grad_scaler = self.adapter.create_grad_scaler(self.config)

        # Sharding
        if self.config.sharding:
            self.model, self.optimizer, self.grad_scaler = self.adapter.train_on_sharding(self.config, self.model, self.optimizer, self.grad_scaler)

        self.model = self.adapter.model_to_ddp(self.config, self.model)



    # @profile(precision=4, stream=open("memory_profiler_train_ckpt.log", "w+"))
    def _init_model(self, model):
        checkpoint_path = os.path.join(self.config.base_path, self.config.data_dir, self.config.init_checkpoint)
        state_dict = paddle.load(checkpoint_path, return_numpy=True)
        
        def convert_state_dict_dtype(state_dict, model_to_load):
            if self.config.fp16:
                dtype = "float16"
            else:
                dtype = "float32"
            # convert the dtype of state dict
            for k, v in model_to_load.state_dict().items():
                # if not isinstance(v, np.ndarray):
                #     dtype = str(v.dtype)[dtype_prefix_len:]
                if k in state_dict:
                    if paddle.in_dynamic_mode():
                        if isinstance(state_dict[k], np.ndarray):
                            state_dict[k] = state_dict[k].astype(dtype)
                        else:
                            state_dict[k] = paddle.cast(state_dict[k], dtype)
        
        convert_state_dict_dtype(state_dict, model)
        model.set_state_dict(state_dict)

        return model

    # @profile(precision=4, stream=open("memory_profiler_train_epoch.log", "w+"))
    def train_one_epoch(self, dataloader, loss):
        state = self.training_state
        driver = self.driver

        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()
        for _, inputs in enumerate(dataloader):
            state.global_steps += 1
            state.num_trained_samples = state.global_steps * dist_paddle.global_batch_size(self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            tr_loss_step = self.train_one_step(inputs)
        
            state.loss = tr_loss_step
            self.tr_loss += tr_loss_step

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (
                    dist_paddle.global_batch_size(self.config) *
                    self.config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second
            
            self.global_steps = state.global_steps
            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_avg_loss = self.evaluator.evaluate(self, self.config)
                eval_end = time.time()
                eval_result = dict(
                    global_steps=state.global_steps,
                    eval_loss=state.eval_avg_loss,
                    time=eval_end - eval_start)
            
            end_training = self.detect_training_status(state)

            
            if state.global_steps % self.config.log_freq == 0:
                if self.config.world_size <= 1:
                    state.loss = self.tr_loss / self.config.log_freq
                else:
                    tr_log_losses = []
                    dist_paddle.all_gather(tr_log_losses, self.tr_loss)
                    tr_log_losses = [t if len(t.shape) > 0 else t.reshape_([-1]) for t in tr_log_losses]
                    concat = paddle.concat(tr_log_losses, axis=0)
                    state.loss = concat.mean() / self.config.log_freq
                self.tr_loss = paddle.to_tensor(0.0)

            step_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END,
                         message=step_info,
                         step=state.global_steps,
                         loss=state.loss)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)
            
            if end_training:
                break

        driver.event(Event.EPOCH_END, state.epoch)

    # @profile(precision=4, stream=open("memory_profiler_train_step.log", "w+"))
    def train_one_step(self, inputs):
        self.model.train()
        state = self.training_state
        
        with self.adapter.autocast_smart_context_manager(self.config):
            outputs = self.model(**inputs)
            loss_step = outputs[0]

        self.adapter.backward(self.config, state.global_steps, loss_step,
                              self.optimizer, self.lr_scheduler, 
                              self.do_grad_scaling, self.grad_scaler)
        print(inputs, outputs, loss_step)

        return loss_step

    def detect_training_status(self, state: TrainingState):
        if state.global_steps >= self.config.max_steps or state.num_trained_samples >= self.config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            self.config.data_dir is not None,
            state.num_trained_samples >= self.config.eval_iter_start_samples,
            self.config.eval_steps > 0,
            state.global_steps > 1,
            state.global_steps % self.config.eval_steps == 0
        ])
        return do_eval or state.global_steps >= self.config.max_steps
