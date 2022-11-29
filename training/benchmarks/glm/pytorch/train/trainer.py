import torch
from torch.types import Device
import os
import time
import math

import utils
from model import create_model
from schedulers import create_scheduler
from utils import main_proc_print
import pickle

import config
from train.driver import Event
from .driver import distributed, Driver
from train.evaluator import Evaluator
from train.training_state import TrainingState


def process_batch(batch, device):
    """Process batch and produce inputs for the model."""
    batch = {t: batch[t].to(device) for t in batch if t != 'answer_idx'}

    return batch


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.overflow_buf = None

    def init(self):
        self.model = create_model(config)
        self.model = self._init_model(self.model, self.config, self.device)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)
        self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        if self.config.fp16 and self.optimizer is not None:
            self.optimizer._model_params_to_master_params()
        self.grad_scaler = self.adapter.create_grad_scaler()

    def _init_model(self, model, args, device):
        checkpoint_name = config.init_checkpoint
        print('checkpoint_name', checkpoint_name)
        print('global rank {} is loading pretrained model {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        # model = model.module

        # Model.
        def extend_embedding_weights(state_weights, model_weights):
            original_length = state_weights.shape[0]
            assert original_length <= args.max_position_embeddings + 1
            new_weights = model_weights.clone()
            new_weights[:original_length] = state_weights
            return new_weights

        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd['module'][
                "transformer.position_embeddings.weight"]
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd['module'][
                    "transformer.position_embeddings.weight"] = extend_embedding_weights(
                        position_weights,
                        model.state_dict()
                        ["transformer.position_embeddings.weight"].data)
                main_proc_print(
                    f"Extend position embedding to {args.max_position_embeddings + 1}"
                )
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd['module'][
                "transformer.block_position_embeddings.weight"]
            if args.max_position_embeddings + 1 > block_position_weights.shape[
                    0]:
                sd['module'][
                    "transformer.block_position_embeddings.weight"] = extend_embedding_weights(
                        block_position_weights,
                        model.state_dict()
                        ["transformer.block_position_embeddings.weight"].data)
                main_proc_print(
                    f"Extend block position embedding to {args.max_position_embeddings + 1}"
                )
        missing_keys, unexpected_keys = model.load_state_dict(sd['module'],
                                                              strict=False)
        if missing_keys or unexpected_keys:
            main_proc_print(
                f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}"
            )
        model = model.to(device)
        return model

    def train_one_epoch(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()
        epoch_start_num_sample = state.num_trained_samples

        for batch_idx, batch in enumerate(dataloader):

            state.global_steps += 1
            # TODO: Maybe we should update num_trained_samples after all epochs.
            state.num_trained_samples = state.global_steps * \
                utils.global_batch_size(self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch)

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (
                    utils.global_batch_size(self.config) *
                    self.config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second

            loss_scale = self.optimizer.loss_scaler.loss_scale
            other_state['loss_scale'] = loss_scale

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_accuracy = self.evaluator.evaluate(self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_accuracy=state.eval_accuracy,
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

        epoch_start_num_sample += len(dataloader.dataset)
        state.num_trained_samples = epoch_start_num_sample

        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, batch):
        data = process_batch(batch, self.config.device)
        state = self.training_state

        # self.training_event.on_step_begin(state.global_steps)
        self.model.train()

        lm_loss, _ = self.forward(data)
        lm_loss /= self.config.gradient_accumulation_steps
        reduced_loss = lm_loss.detach().clone().view(1)
        torch.distributed.all_reduce(reduced_loss.data)
        reduced_loss.data = reduced_loss.data / (utils.get_world_size())

        state.loss = lm_loss
        #lm_loss.backward()
        #self.optimizer.step()
        self.adapter.backward(state.global_steps, lm_loss, reduced_loss,
                              self.optimizer, self.lr_scheduler, self.model)
        #self.adapter.backward(state.global_steps, state.loss, self.optimizer)
        #self.adapter.backward(state.global_steps, reduced_loss, self.optimizer)
        #self.adapter.backward(state.global_steps, reduced_loss, self.optimizer, self.lr_scheduler)
        # self.training_event.on_backward(
        #     state.global_steps, lm_loss, reduced_loss, self.optimizer, self.lr_scheduler)
        #self.lr_scheduler.step()
        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer, self.grad_scaler)
        #self.lr_scheduler.step()

    def detect_training_status(self, state):
        config = self.config
        if state.eval_accuracy >= config.target_accuracy:
            state.converged_success()

        if state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state):
        config = self.config
        do_eval = all([
            config.eval_data is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps %
            math.ceil(config.eval_interval_samples /
                      utils.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination

    def forward(self, batch):
        data = batch
        tokens, labels, position_ids, attention_mask = data['text'], data[
            'label'], data['position'], data['mask']
        target_ids, logit_mask = data['target'], data['logit_mask']

        result = self.model(tokens, position_ids, attention_mask, target_ids,
                            logit_mask)
        logits, *mems = result

        loss_mask = data["loss_mask"]
        logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.contiguous().float(), labels)

        return loss, mems


# class Trainer:
#     def __init__(self, training_event, evaluator, training_state, config):
#         self.training_event = training_event
#         self.evaluator = evaluator
#         self.training_state = training_state
#         self.config = config

#         self.optimizer = None
#         self.model = None
#         self.lr_scheduler = None

#     def init(self):
#         self.model = create_model(self.config)
#         self._init_model(self.model, self.config)
#         self.model = self.training_event.convert_model(self.model)
#         self.model = self.training_event.model_to_fp16(
#             self.model)
#         self.optimizer = self.training_event.create_optimizer(self.model)
#         self.model = self.training_event.model_to_ddp(self.model)
#         self.lr_scheduler = create_scheduler(self.optimizer, self.config)
#         if self.config.fp16 and self.optimizer is not None:
#             self.optimizer._model_params_to_master_params()

#     def _init_model(self, model, args):
#         checkpoint_name = 'tmp/pytorch_model.bin'
#         print('global rank {} is loading pretrained model {}'.format(
#             torch.distributed.get_rank(), checkpoint_name))
#         # Load the checkpoint.
#         sd = torch.load(checkpoint_name, map_location='cpu')
#         # model = model.module

#         # Model.
#         def extend_embedding_weights(state_weights, model_weights):
#             original_length = state_weights.shape[0]
#             assert original_length <= args.max_position_embeddings + 1
#             new_weights = model_weights.clone()
#             new_weights[:original_length] = state_weights
#             return new_weights

#         if "transformer.block_position_embeddings.weight" in sd["module"]:
#             position_weights = sd['module']["transformer.position_embeddings.weight"]
#             if args.max_position_embeddings + 1 > position_weights.shape[0]:
#                 sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
#                     position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
#                 main_proc_print(
#                     f"Extend position embedding to {args.max_position_embeddings + 1}")
#         if "transformer.block_position_embeddings.weight" in sd["module"]:
#             block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
#             if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
#                 sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
#                     block_position_weights,
#                     model.state_dict()["transformer.block_position_embeddings.weight"].data)
#                 main_proc_print(
#                     f"Extend block position embedding to {args.max_position_embeddings + 1}")
#         missing_keys, unexpected_keys = model.load_state_dict(
#             sd['module'], strict=False)
#         if missing_keys or unexpected_keys:
#             main_proc_print(
#                 f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")

#     def train_one_epoch(self, dataloader):
#         state = self.training_state
#         training_event = self.training_event

#         training_event.on_epoch_begin(state.epoch)

#         step_start_time = time.time()
#         epoch_start_num_sample = state.num_trained_samples

#         for batch_idx, batch in enumerate(dataloader):

#             state.global_steps += 1
#             # 考虑到最后一个batch可能会小于设定的batch size，所以 epoch完成之后，需要更新下 num_trained_samples
#             state.num_trained_samples = state.global_steps * \
#                 utils.global_batch_size(self.config)

#             self.train_one_step(batch)

#             other_state = dict()
#             if state.global_steps % self.config.gradient_accumulation_steps == 0:
#                 step_end_time = time.time()
#                 step_total_time = step_end_time - step_start_time
#                 step_start_time = step_end_time
#                 sequences_per_second = (utils.global_batch_size(
#                     self.config) * self.config.gradient_accumulation_steps) / step_total_time
#                 other_state["seq/s"] = sequences_per_second

#             loss_scale = self.optimizer.loss_scaler.loss_scale
#             other_state['loss_scale'] = loss_scale

#             eval_result = None
#             if self.can_do_eval(state):
#                 eval_start = time.time()
#                 state.eval_accuracy = self.evaluator.evaluate(self)
#                 eval_end = time.time()
#                 eval_result = dict(global_steps=state.global_steps,
#                                    eval_accuracy=state.eval_accuracy,
#                                    time=eval_end - eval_start)

#             end_training = self.detect_training_status(state)

#             step_info = state.to_dict(**other_state)
#             self.training_event.on_step_end(state.global_steps, result=step_info)

#             if eval_result is not None:
#                 self.training_event.on_evaluate(eval_result)

#             if end_training:
#                 break

#         epoch_start_num_sample += len(dataloader.dataset)
#         state.num_trained_samples = epoch_start_num_sample

#         training_event.on_epoch_end(state.epoch)

#     def train_one_step(self, batch):
#         data = process_batch(batch, self.config.device)
#         state = self.training_state

#         self.training_event.on_step_begin(state.global_steps)
#         self.model.train()

#         lm_loss, _ = self.forward(data)
#         lm_loss /= self.config.gradient_accumulation_steps
#         reduced_loss = lm_loss.detach().clone().view(1)
#         torch.distributed.all_reduce(reduced_loss.data)
#         reduced_loss.data = reduced_loss.data / (utils.get_world_size())

#         state.loss = reduced_loss
#         self.training_event.on_backward(
#             state.global_steps, lm_loss, reduced_loss, self.optimizer, self.lr_scheduler)

#         self.training_event.on_step_end(state.global_steps)

#     def forward(self, batch):
#         data = batch
#         tokens, labels, position_ids, attention_mask = data[
#             'text'], data['label'], data['position'], data['mask']
#         target_ids, logit_mask = data['target'], data['logit_mask']

#         result = self.model(tokens, position_ids, attention_mask,
#                             target_ids, logit_mask)
#         logits, *mems = result

#         loss_mask = data["loss_mask"]
#         logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)

#         loss_func = torch.nn.CrossEntropyLoss()
#         loss = loss_func(logits.contiguous().float(), labels)

#         return loss, mems

#     def forward_step(self, batch, model, mems):
#         data = batch
#         tokens, labels, position_ids = data['text'], data['label'], data['position']
#         attention_mask = data['mask']
#         target_ids, logit_mask = data['target'], data['logit_mask']
#         result = model(tokens, position_ids, attention_mask,
#                        target_ids, logit_mask)
#         logits, *mems = result

#         loss_mask = data["loss_mask"]
#         logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)

#         loss_func = torch.nn.CrossEntropyLoss()
#         loss = loss_func(logits.contiguous().float(), labels)

#         return loss, mems

#     def can_do_eval(self, state):
#         config = self.config
#         do_eval = all([
#             config.eval_data is not None,
#             state.num_trained_samples >= config.eval_iter_start_samples,
#             state.global_steps % math.ceil(config.eval_interval_samples / utils.global_batch_size(config)) == 0,
#             config.eval_interval_samples > 0,
#             state.global_steps > 1,
#         ])

#         return do_eval or state.num_trained_samples >= config.max_samples_termination

#     def detect_training_status(self, state):
#         config = self.config
#         if state.eval_accuracy >= config.target_accuracy:
#             state.converged_success()

#         if state.num_trained_samples > config.max_samples_termination:
#             state.end_training = True

#         return state.end_training
