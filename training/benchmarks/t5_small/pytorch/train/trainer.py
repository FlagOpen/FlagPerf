# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import torch
import torch.utils.data
from torch.types import Device
import os
import sys
import torch.distributed as dist
from accelerate import Accelerator

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

    def init(self, train_dataloader, eval_dataloader):
        device = torch.device(self.config.device)
        dist_pytorch.main_proc_print("Init progress:")
        self.model, self.model_config, self.tokenizer = create_model(
            self.config)
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, train_dataloader,
                                             self.config)

        self.accelerator = Accelerator()
        self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader,
            self.lr_scheduler)

        return train_dataloader, eval_dataloader

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def train_one_epoch(self, train_dataloader, eval_dataloader):

        model = self.model
        optimizer = self.optimizer
        data_loader = train_dataloader
        device = self.device
        epoch = self.training_state.epoch
        print("Epoch " + str(epoch + 1))

        model.train()
        noeval_start_time = time.time()

        for step, batch in enumerate(data_loader):
            batch = self.process_batch(batch, device)

            pure_start_time = time.time()

            outputs = model(**batch)
            loss = outputs.loss

            self.accelerator.backward(loss)
            optimizer.step()
            self.lr_scheduler.step()
            optimizer.zero_grad()

            if step % self.config.log_freq == 0:
                print("Train Step " + str(step) + "/" + str(len(data_loader)) +
                      ", Loss : " + str(float(loss)))

            self.training_state.purecomputetime += time.time(
            ) - pure_start_time

        self.training_state.noevaltime += time.time() - noeval_start_time

        eval_result = self.evaluate(self.model,
                                    eval_dataloader,
                                    device=self.device)

        state = self.training_state
        config = self.config

        state.rouge1, state.rouge2, state.rougeL, state.rougeLsum = eval_result.values(
        )
        if state.rouge1 >= config.target_rouge1:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_rouge1: {state.rouge1}, target_rouge1: {config.target_rouge1}"
            )
            state.converged_success()

        if epoch + 1 >= config.max_epoch:
            state.end_training = True
        state.num_trained_samples += len(data_loader.dataset)

    def evaluate(self, model, data_loader, device):
        self.model.eval()
        self.evaluator.reset()
        for step, batch in enumerate(data_loader):
            if step % self.config.log_freq == 0:
                print("Eval Step " + str(step) + "/" + str(len(data_loader)))
            batch = self.process_batch(batch, device)
            input_ids, labels = batch['input_ids'], batch['labels']

            # https://github.com/huggingface/transformers/blob/v4.31.0/examples/pytorch/summarization/run_summarization.py#L178
            # https://github.com/huggingface/transformers/blob/v4.31.0/examples/pytorch/summarization/run_summarization.py#L651C2-L655
            # according to huggingface run_summarization.py, max_length is 128, num_beams is 1
            def _unwrap_model(model):
                if hasattr(model, "module"):
                    return _unwrap_model(model.module)
                else:
                    return model

            model = _unwrap_model(model)
            output = model.generate(input_ids,
                                    max_length=128,
                                    num_beams=self.model_config.num_beams)
            self.evaluator.add_batch(self.tokenizer, output, labels)

        result = self.evaluator.compute_acc()
        dist_pytorch.main_proc_print(result)

        return result
