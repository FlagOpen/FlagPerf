import torch
from torch.types import Device
import torch.distributed as dist
import os
import sys
import time
import math

from models import create_model
from schedulers import create_scheduler

from train.evaluator import Evaluator
from train.training_state import TrainingState

import config

from timm.utils import accuracy, AverageMeter
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config, len_train_dataloader):
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
        self.len_train_dataloader = len_train_dataloader
        # self.global_batch_size = None
        # self.overflow_buf = None

    def init(self):
        self.model = create_model(config)
        self.model = self.adapter.convert_model(self.model)
        # self.model = self.adapter.model_to_fp16(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)
        self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config, self.len_train_dataloader)
        # self.grad_scaler = self.adapter.create_grad_scaler()
        self.loss_scaler = NativeScalerWithGradNormCount()


    # 接入perf的日志体系，dirver events
    def train_one_epoch(self, criterion, dataloader, epoch, mixup_fn):
        config = self.config
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss_scaler = self.loss_scaler
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

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples)
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
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
            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                driver.event(Event.EPOCH_END,state.loss)
                # logger.info(
                #     f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                #     f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                #     f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                #     f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                #     f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                #     f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                #     f'mem {memory_used:.0f}MB')
            # epoch_time = time.time() - start
            # logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

            # 不需要每个step去做eval，需要设置间隔
            acc1, acc5, loss = self.evaluator.evaluate(config, model)
            max_accuracy = max(max_accuracy, acc1)
            
            state.eval_acc1, state.eval_acc5, state.eval_loss = acc1, acc5, loss
            state.max_accuracy = max_accuracy
            driver.event(Event.EVALUATE, state.eval_acc1, state.eval_acc5, state.eval_loss, state.max_accuracy)
            
            end_training = self.detect_training_status(state)

            if end_training:
                break

        # epoch_start_num_sample += len(dataloader.dataset)
        # state.num_trained_samples = epoch_start_num_sample

        # self.lr_scheduler.step()
        # driver.event(Event.EPOCH_END, state.epoch)


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
