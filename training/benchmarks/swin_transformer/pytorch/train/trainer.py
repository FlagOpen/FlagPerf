import torch
from torch.types import Device
import torch.distributed as dist
import os
import sys
import time
import math
import datetime

from models import create_model
from schedulers import create_scheduler

from train.evaluator import Evaluator
from train.training_state import TrainingState

import config
from utils.logger import create_logger
from timm.utils import accuracy, AverageMeter
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


class Trainer:

    def __init__(self, driver: Driver, training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.training_state = training_state
        self.device = device
        self.config = config

    # 接入perf的日志体系，dirver events
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
        for idx, (samples, targets) in enumerate(dataloader):
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
            # state.loss, state.acc1, state.acc5 = total.tolist()
            # driver.event(Event.EPOCH_END, state.loss)
            if idx % config.print_freq == 0 and config.local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                etas = batch_time.avg * (num_steps - idx)
                print(
                    f'Train: [{epoch}/{config.train_epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t')
        epoch_time = time.time() - start
        if config.local_rank == 0:
            print("EPOCH {} training takes {}".format(epoch, datetime.timedelta(seconds=int(epoch_time))))


    def detect_training_status(self, state):
        config = self.config
        if state.eval_acc1 >= config.target_acc1:
            state.converged_success()

        # if state.num_trained_samples > config.max_samples_termination:
        #     state.end_training = True

        return state.end_training


    def process_batch(self, batch, device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch
