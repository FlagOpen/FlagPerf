import os
import sys
import time
import datetime

import torch
from torch.types import Device
from timm.utils import accuracy, AverageMeter

from driver import Driver, Event, dist_pytorch
from train.training_state import TrainingState

class Trainer:

    def __init__(self, driver: Driver, training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.training_state = training_state
        self.device = device
        self.config = config

    def train_one_epoch(self, model, criterion, dataloader, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler):
        config = self.config
        driver = self.driver
        state = self.training_state
        
        model.train()
        optimizer.zero_grad()

        num_steps = len(dataloader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()

        start = time.time()
        end = time.time()
        step_start_time = time.time()
        for idx, (samples, targets) in enumerate(dataloader):
            state.global_steps += 1
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            with torch.cuda.amp.autocast(enabled=config.amp_enable):
                outputs = model(samples)
            loss = criterion(outputs, targets)
            loss = loss / config.train_accumulation_steps

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train_clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.train_accumulation_steps == 0)
            if (idx + 1) % config.train_accumulation_steps == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.train_accumulation_steps)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()
            state.loss = loss_meter.val
            
            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                images_per_second = (
                    dist_pytorch.global_batch_size(self.config) *
                    self.config.gradient_accumulation_steps) / step_total_time
                other_state["img/s"] = images_per_second
        
            other_state['loss_scale'] = scaler_meter.val
            other_state['lr'] = optimizer.param_groups[0]['lr']
            other_state['weight_decay'] = optimizer.param_groups[0]['weight_decay']
            
            step_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END,
                         message=step_info,
                         step=state.global_steps,
                         loss=state.loss)         
            
        epoch_time = time.time() - start
        if config.local_rank == 0:
            print("EPOCH {} training takes {}".format(epoch, datetime.timedelta(seconds=int(epoch_time))))


    def detect_training_status(self, state):
        config = self.config
        if state.eval_acc1 >= config.target_acc1:
            state.converged_success()

        return state.end_training

