import math
import time

import torch
from torch.types import Device

from model import create_model
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.metrics import average_corpus_level
from train.training_state import TrainingState
from model.losses.cross_entropy import cross_entropy
from model.fp16 import FP16_Module
from driver import Driver, Event, dist_pytorch


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

        self.lr_scheduler = create_scheduler(self.optimizer, self.config)

    def _init_model(self, model, device):
        checkpoint = torch.load(self.config.init_checkpoint,
                                map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        model.load_state_dict(checkpoint, strict=True)

        model = model.to(device)
        return model

    def train_one_epoch(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()

        for _, data in enumerate(dataloader):
            no_eval_start_time = time.time()
            batch, no_model_batch = data[0], data[1]

            state.global_steps += 1
            state.num_trained_samples = state.global_steps * dist_pytorch.global_batch_size(
                self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch, no_model_batch)
            self.training_state.no_eval_time += time.time(
            ) - no_eval_start_time
            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (
                    dist_pytorch.global_batch_size(self.config) *
                    self.config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_avg_loss, state.eval_embedding_average = self.evaluator.evaluate(
                    self)
                eval_end = time.time()
                eval_result = dict(
                    global_steps=state.global_steps,
                    eval_loss=state.eval_avg_loss,
                    eval_embedding_average=state.eval_embedding_average,
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

        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, batch, no_model_batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(self.device)

        pure_compute_start_time = time.time()
        state = self.training_state
        self.model.train()

        output = self.model(**batch)
        labels = no_model_batch["labels"]

        #losses 的形状：[b,s]
        losses = cross_entropy(output.contiguous().float(), labels)
        loss_mask = no_model_batch["loss_mask"].view(-1)
        #loss 为标量
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        state.loss = loss

        self.adapter.backward(self.config, state.global_steps, state.loss,
                              self.optimizer)
        self.training_state.pure_compute_time += time.time(
        ) - pure_compute_start_time

        # calculate output
        preds = torch.argmax(output, -1)

        if not hasattr(self.model, "module"):
            embeddings = self.model.word_embeddings.weight
        elif isinstance(self.model.module, FP16_Module):
            embeddings = self.model.module.module.word_embeddings.weight
        else:
            embeddings = self.model.module.word_embeddings.weight

        #embedding_average 形状是[batch_size]
        embedding_average = average_corpus_level(
            preds.cpu().detach(),
            labels.cpu().detach(),
            embeddings.cpu().detach(),
            no_model_batch["loss_mask"].cpu().detach())
        state.embedding_average = float(embedding_average.mean)

        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer)
        self.lr_scheduler.step()

    def detect_training_status(self, state: TrainingState):
        if state.eval_embedding_average >= self.config.target_embedding_average:
            state.converged_success()

        if state.global_steps >= self.config.max_steps or state.num_trained_samples >= self.config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            self.config.data_dir is not None,
            state.num_trained_samples >= self.config.eval_iter_start_samples,
            self.config.eval_interval_samples > 0,
            state.global_steps > 1,
            state.global_steps %
            math.ceil(self.config.eval_interval_samples /
                      dist_pytorch.global_batch_size(self.config)) == 0,
        ])

        return do_eval or state.global_steps >= self.config.max_steps
