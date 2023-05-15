import os
import math
import time
import torch
import torch.distributed as dist

from torch.types import Device
from model import create_model
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState
import config
from driver import Driver, Event, dist_pytorch


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Trainer:
    """Trainer"""

    def __init__(
        self,
        driver: Driver,
        criterion,
        adapter,
        evaluator: Evaluator,
        training_state: TrainingState,
        device: Device,
        config,
    ):
        super(Trainer, self).__init__()
        self.driver = driver
        self.criterion = criterion
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
        self.best_acc1 = 0
        self.eval_accuracy = 0

    def init(self):
        """init"""
        self.model = create_model(self.config)
        self.model = self.adapter.convert_model(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.model = self._init_model(self.model, self.device)

    def _init_model(self, model, device):
        # resume from checkpoint
        if self.config.init_checkpoint:
            if os.path.isfile(config.init_checkpoint):
                loc = "cpu"
                if torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu
                    loc = f"cuda:{config.local_rank}"
                checkpoint = torch.load(config.init_checkpoint,
                                        map_location=loc)
                self.training_state.epoch = checkpoint["epoch"]
                self.training_state.best_acc1 = checkpoint["best_acc1"]
                arch = checkpoint["arch"]
                # correctly resume training from checkpoint for single-card training
                if not self.config.distributed:
                    model = torch.nn.DataParallel(model)

                model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                print(
                    f"=>model({arch}).loaded checkpoint '{config.init_checkpoint}' (epoch {checkpoint['epoch']})"
                )
        else:
            print(f"=> no checkpoint found at '{config.init_checkpoint}'")

        model = model.to(device)
        return model

    def train_one_epoch(self, train_loader):
        """train_one_epoch"""

        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()
        epoch_start_num_sample = state.num_trained_samples

        for _, batch in enumerate(train_loader):

            state.global_steps += 1

            state.num_trained_samples = (
                state.global_steps *
                dist_pytorch.global_batch_size(self.config))

            self.train_one_step(batch)

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (
                    dist_pytorch.global_batch_size(self.config) *
                    self.config.gradient_accumulation_steps) / step_total_time
                other_state["img/s"] = round(sequences_per_second, 1)

            if hasattr(self.optimizer, 'loss_scaler'):
                loss_scale = self.optimizer.loss_scaler.loss_scale
                other_state['loss_scale'] = loss_scale

            eval_result = None

            if self.can_do_eval(state):
                eval_start = time.time()
                # evaluate on validation set
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
            driver.event(
                Event.STEP_END,
                message=step_info,
                step=state.global_steps,
                loss=state.loss,
            )

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if end_training:
                print(f"end_training: {end_training}")
                break

        epoch_start_num_sample += len(train_loader.dataset)
        state.num_trained_samples = epoch_start_num_sample

        self.lr_scheduler.step()
        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, batch):
        """train_one_step"""
        # move data to the same device as model
        batch = self.process_batch(batch, self.config.device)
        state = self.training_state
        self.model.train()
        state.loss, state.acc1, state.acc5 = self.forward(batch)
        self.adapter.backward(state.global_steps, state.loss, self.optimizer)
        if dist.is_available() and dist.is_initialized():
            total = torch.tensor([state.loss, state.acc1, state.acc5],
                                 dtype=torch.float32,
                                 device=self.config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            total = total / dist.get_world_size()
            state.loss, state.acc1, state.acc5 = total.tolist()
        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer, self.grad_scaler)

    def can_do_eval(self, state):
        """can_do_eval"""

        config = self.config
        do_eval = all([
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps %
            math.ceil(config.eval_interval_samples /
                      dist_pytorch.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination

    def detect_training_status(self, state):
        """detect_training_status"""
        config = self.config
        if state.eval_acc1 >= config.target_acc1:
            print(
                f"converged_success. eval_acc1: {state.eval_acc1}, target_acc1: {config.target_acc1}"
            )
            state.converged_success()

        if state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def forward(self, batch):
        """forward"""
        images, target = batch
        output = self.model(images)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5

    def inference(self, batch):
        """inference"""
        self.model.eval()
        output = self.forward(batch)
        return output

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch
