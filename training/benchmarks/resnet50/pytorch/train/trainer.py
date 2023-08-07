# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import time
import torch
import torch.utils.data
import torchvision
from torch.types import Device
import os
import sys
import torch.distributed as dist

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

    def init(self):
        torch.set_num_threads(1)
        device = torch.device(self.config.device)
        dist_pytorch.main_proc_print("Init progress:")
        self.model = create_model()
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)

        self.scaler = self.adapter.create_grad_scaler()

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch

    def train_one_epoch(self, train_dataloader, eval_dataloader):

        model = self.model
        optimizer = self.optimizer
        data_loader = train_dataloader
        device = self.device
        epoch = self.training_state.epoch
        scaler = self.scaler
        print("Epoch " + str(epoch + 1))
        if self.config.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        model.train()
        noeval_start_time = time.time()

        lr_scheduler = None

        for step, batch in enumerate(data_loader):

            batch = self.process_batch(batch, device)

            pure_start_time = time.time()
            optimizer.zero_grad()

            images, target = batch
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(images)

                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(images)

                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if step % self.config.log_freq == 0:
                print("Train Step " + str(step) + "/" + str(len(data_loader)) +
                      ", Loss : " + str(float(loss)))

            self.training_state.purecomputetime += time.time(
            ) - pure_start_time

        self.lr_scheduler.step()
        self.training_state.noevaltime += time.time() - noeval_start_time

        acc1 = self.evaluate(self.model, eval_dataloader, device=self.device)

        state = self.training_state
        config = self.config

        state.acc1 = acc1
        if state.acc1 >= config.target_acc1:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_mAP: {state.acc1}, target_mAP: {config.target_acc1}"
            )
            state.converged_success()

        if epoch + 1 >= config.max_epoch:
            state.end_training = True
        state.num_trained_samples += len(data_loader.dataset)

    @torch.no_grad()
    def evaluate(self, model, data_loader, device):
        acc1_total = 0.0
        steps = 0
        for step, batch in enumerate(data_loader):
            if step % self.config.log_freq == 0:
                print("Eval Step " + str(step) + "/" + str(len(data_loader)))
            batch = self.process_batch(batch, device)
            images, target = batch
            output = model(images)
            acc1, acc5 = self.evaluator.accuracy(output, target, (1, 5))
            acc1_total += acc1
            steps += 1
        acc1 = torch.tensor([acc1_total], dtype=torch.float32, device=device)
        world_size = 1
        if self.config.distributed:
            dist.all_reduce(acc1, dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        acc1 = acc1 / (steps * world_size)
        print("Eval Acc1: " + str(float(acc1)) + "%")
        return float(acc1)
