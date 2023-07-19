import math
import time
import os
import sys

import torch
from torch.cuda.amp import GradScaler
from torch.types import Device

import config
import utils
from dataloaders.dataset import exchange_padding_fast
from model import create_model
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState
from utils.checkpoint import remap_segmented_model_parameters

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import Driver, Event, dist_pytorch


class Trainer():

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, grad_scaler: GradScaler,
                 device: Device):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = grad_scaler

        self.device = device
        self.optimizer = None
        self.bert_config = None
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None

    def init(self):
        self.bert_config, self.model = create_model(config)
        self.model = self._init_model(self.model, self.device)
        self.model = self.adapter.convert_model(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model)
        self.model, self.optimizer = self.adapter.model_to_fp16(
            self.model, self.optimizer)
        self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer)

    def _init_model(self, model, device):
        checkpoint = torch.load(config.init_checkpoint, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        checkpoint_remapped = remap_segmented_model_parameters(checkpoint)

        model.load_state_dict(checkpoint_remapped, strict=True)
        import copy

        model = model.to(device)
        return model

    def train_one_epoch(self, dataloader):
        state = self.training_state
        adapter = self.adapter

        #adapter.on_epoch_begin(state.epoch)
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        no_eval_start = time.time()
        for dataloader_idx, batch_idx, batch in dataloader.iter_batchs():
            pure_compute_start = time.time()

            state.num_trained_samples = state.global_steps * utils.global_batch_size(
                config)

            state.global_steps += 1
            state.iter_dataloader_idx = dataloader_idx
            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch_idx, batch)
            
            state.pure_compute_time += time.time() - pure_compute_start
            state.no_eval_time += time.time() - no_eval_start


            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_loss, state.eval_mlm_accuracy = self.evaluator.evaluate(
                    self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_loss=state.eval_loss,
                                   eval_mlm_accuracy=state.eval_mlm_accuracy,
                                   time=eval_end - eval_start)

            end_training = self.detect_training_status(state)

            driver.event(Event.STEP_END,
                         message={"loss":float(state.loss)},
                         step=state.global_steps,
                         loss=state.loss)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if end_training:
                break
            no_eval_start = time.time()
        driver.event(Event.EPOCH_END, state.epoch)
        #adapter.on_epoch_end(state.epoch)

    def train_one_step(self, batch_idx, batch):
        if config.exchange_padding == True:
            batch = [
                t.to(self.device, non_blocking=True, dtype=torch.int16)
                for t in batch
            ]
            batch = exchange_padding_fast(self.device, config.train_batch_size,
                                          *batch)
        else:
            batch = [t.to(self.device, non_blocking=True) for t in batch]

        state = self.training_state
        self.model.train()
        state.loss, state.mlm_acc, _ = self.forward(batch)
        self.adapter.backward(state.global_steps, state.loss, self.optimizer,
                              self.grad_scaler)
        self.lr_scheduler.step()

    def detect_training_status(self, state: TrainingState):
        if state.eval_mlm_accuracy >= config.target_mlm_accuracy:
            state.converged_success()

        if state.global_steps >= config.max_steps or state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            config.eval_dir is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps % config.eval_step == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.global_steps >= config.max_steps

    def forward(self, batch):
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        loss, mlm_acc, num_valid = self.model(input_ids, segment_ids,
                                              input_mask, masked_lm_labels,
                                              next_sentence_labels)
        return loss, mlm_acc, num_valid

    def inference(self, batch):
        self.model.eval()
        return self.forward(batch)
