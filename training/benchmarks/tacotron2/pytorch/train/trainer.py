import torch
from torch.types import Device
import os
import sys
import numpy as np

from model import create_model, create_model_config
from model.loss.loss_function import get_loss_function
from model.data.data_function import batch_to_gpu
from optimizers import create_optimizer
from .utils import reduce_tensor

from train.evaluator import Evaluator
from train.training_state import TrainingState
import config

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 world_size, train_dataloader):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.model_config = None
        self.evaluator = evaluator
        self.global_batch_size = None
        self.criterion = None
        self.world_size = world_size
        self.train_dataloader = train_dataloader

    def init(self):
        self.model_config = create_model_config(config)
        self.model = create_model(config)
        self.model = self.adapter.model_to_ddp(self.model, self.config)
        self._init_model()

        self.criterion = get_loss_function()
        self.optimizer = create_optimizer(self.model, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler(self.config)
        torch.backends.cudnn.enabled = self.config.cudnn_enabled
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark

    def _init_model(self):
        self.model.train()

    def train_one_epoch(self, train_dataloader):
        state = self.training_state
        driver = self.driver

        if self.config.local_rank == 0:
            state.epoch += 1

        driver.event(Event.EPOCH_BEGIN, state.epoch)

        torch.cuda.synchronize()

        if self.config.distributed:
            self.train_dataloader.sampler.set_epoch(state.epoch)

        epoch_start_num_sample = state.num_trained_samples

        for batch in train_dataloader:
            self.train_one_step(batch)

        torch.cuda.synchronize()

        val_loss, _ = self.evaluator.evaluate(self)
        state.val_loss = val_loss

        epoch_start_num_sample += len(train_dataloader.dataset)
        state.num_trained_samples = epoch_start_num_sample
        epoch_data = {"val_loss": val_loss, "epoch":state.epoch, "global_steps": state.global_steps}
        driver.event(Event.EPOCH_END, state.epoch, message=epoch_data)

        self.detect_training_status()

    def train_one_step(self, batch):
        driver = self.driver
        state = self.training_state
        args = self.config

        torch.cuda.synchronize()
        adjust_learning_rate(self.training_state.epoch, self.optimizer,
                             args.learning_rate, args.lr_anneal_steps,
                             args.lr_anneal_factor)

        self.model.zero_grad()
        x, y, _ = batch_to_gpu(batch)

        # AMP upstream autocast
        with torch.cuda.amp.autocast(enabled=args.amp):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, self.world_size).item()
        else:
            reduced_loss = loss.item()

        if np.isnan(reduced_loss):
            raise Exception("loss is NaN")

        if args.amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           args.grad_clip_thresh)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           args.grad_clip_thresh)
            self.optimizer.step()

        self.model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        state.train_loss = reduced_loss
        step_info = dict(step=state.global_steps, train_loss=reduced_loss)

        self.training_state.global_steps += 1
        driver.event(Event.STEP_END, state.global_steps, message=step_info)

    def detect_training_status(self):
        config = self.config
        state = self.training_state
        # for loss: the smaller, the better
        if state.val_loss <= config.target_val_loss:
            state.converged_success()

        if state.epoch > config.max_epochs:
            state.end_training = True

        return state.end_training


def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps,
                         anneal_factor):
    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1**(p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor**p)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
