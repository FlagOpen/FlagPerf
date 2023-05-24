import torch
from torch.types import Device
import torch.distributed as dist
import os
import sys
import time
import math

from model import create_model
from schedulers import create_scheduler

from train.evaluator import Evaluator
from train.training_state import TrainingState
from train import utils

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
        self.scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None

    def init(self):
        self.model = create_model(config)
        self.model = self._init_model(self.model, self.config, self.device)
        self.model = self.adapter.convert_model(self.config, self.model)
        self.model = self.adapter.model_to_fp16(self.config, self.model)
        self.optimizer = self.adapter.create_optimizer(self.config, self.model)
        self.model = self.adapter.model_to_ddp(self.config, self.model)

        self.lr_scheduler = create_scheduler(self.config, self.optimizer)
        self.scaler = self.adapter.create_grad_scaler(self.config)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        self.resume()

    def _init_model(self, model, args, device):
        # checkpoint_name = config.init_checkpoint
        # if os.path.isfile(checkpoint_name):
        #     print('checkpoint_name', checkpoint_name)
        #     print('global rank {} is loading pretrained model {}'.format(
        #         dist_pytorch.get_rank(), checkpoint_name))
        #     # Load the checkpoint.
        #     checkpoint = torch.load(checkpoint_name, map_location='cpu')
        #     model.load_state_dict(checkpoint['state_dict'])

        model = model.to(device)
        return model
    
    def resume(self):
        args = self.config
        if args.resume and os.path.isfile(args.resume):
            print('global rank {} is loading checkpoint {}'.format(
                dist_pytorch.get_rank(), args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.training_state.load_state_dict(checkpoint["training_state"])
            self.training_state.epoch += 1
            if self.scaler:
                self.scaler.load_state_dict(checkpoint["scaler"])

    def train_one_epoch(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()
        epoch_start_num_sample = state.num_trained_samples

        if dist.is_available() and dist.is_initialized():
            dataloader.sampler.set_epoch(state.epoch)
        for batch_idx, batch in enumerate(dataloader):

            state.global_steps += 1
            # TODO: Maybe we should update num_trained_samples after all epochs.
            state.num_trained_samples = state.global_steps * \
                dist_pytorch.global_batch_size(self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch)

            other_state = dict()

            step_end_time = time.time()
            step_total_time = step_end_time - step_start_time
            step_start_time = step_end_time
            images_per_second = dist_pytorch.global_batch_size(self.config) / step_total_time
            other_state["img/s"] = images_per_second
            if hasattr(self.optimizer, 'loss_scaler'):
                loss_scale = self.optimizer.loss_scaler.loss_scale
                other_state['loss_scale'] = loss_scale

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_loss, state.eval_acc1, state.eval_acc5 = self.evaluator.evaluate(
                    self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_loss=state.eval_loss,
                                   eval_acc1=state.eval_acc1,
                                   eval_acc5=state.eval_acc5,
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

        self.lr_scheduler.step()
        if self.config.output_dir:
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "training_state": self.training_state.state_dict(),
                #"epoch": epoch,
                #"args": args,
            }
            if self.scaler:
                checkpoint["scaler"] = self.scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(self.config.output_dir, f"model_{self.training_state.epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(self.config.output_dir, "checkpoint.pth"))
        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, batch):
        # move data to the same device as model
        batch = self.process_batch(batch, self.config.device)
        state = self.training_state
        self.model.train()
        state.loss, state.acc1, state.acc5 = self.forward(batch)
        self.adapter.backward(self.config, state.global_steps, state.epoch, state.loss, self.model, self.optimizer, self.scaler)
        if dist.is_available() and dist.is_initialized():
            total = torch.tensor([state.loss, state.acc1, state.acc5],
                                 dtype=torch.float32,
                                 device=self.config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            total = total / dist.get_world_size()
            state.loss, state.acc1, state.acc5 = total.tolist()
        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer, self.scaler)

    def detect_training_status(self, state):
        config = self.config
        if state.eval_acc1 >= config.target_acc1:
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
                      dist_pytorch.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination

    def forward(self, batch):
        images, target = batch
        output = self.model(images)
        loss = self.criterion(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5

    def inference(self, batch):
        self.model.eval()
        output = self.forward(batch)
        return output

    def process_batch(self, batch, device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch
