import torch
from torch.types import Device
import os
import sys
import time
import math

from model import create_model
from schedulers import create_scheduler
from model.loss.loss_function import get_loss_function

from train.evaluator import Evaluator
from train.training_state import TrainingState

import config

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
        self.grad_scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.criterion = None

    def init(self):
        self.model = create_model(config)
        self.criterion = get_loss_function()
        print("============ init ==========")
        print(self.model)
        print("============ init ==========")
        # self.model = self.adapter.model_to_fp16(self.model, self.optimizer)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)
        self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)

        self.grad_scaler = self.adapter.create_grad_scaler(self.config)

    def train_one_epoch(self, train_dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()
        epoch_start_num_sample = state.num_trained_samples

        for batch_idx, batch in enumerate(train_dataloader):
            print(f"batch_idx: {batch_idx}")

            if batch_idx > 10:
                state.converged_success()

            if state.end_training:
                break

        epoch_start_num_sample += len(train_dataloader.dataset)
        state.num_trained_samples = epoch_start_num_sample

        driver.event(Event.EPOCH_END, state.epoch)
