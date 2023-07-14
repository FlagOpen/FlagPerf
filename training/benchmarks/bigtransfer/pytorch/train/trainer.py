import math
import time
import torch
import torch.utils.data
import torchvision
from torch.types import Device
import os
import sys
import numpy as np

from model import create_model
from optimizers import create_optimizer
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
        self.model = create_model(self.config)
        print(f"model: BiT-M-R152x{self.config.model_shard}.npz")
        self.model.load_from(
            np.load(
                f"{self.config.data_dir}{self.config.transfered_weight}BiT-M-R152x{self.config.model_shard}.npz"
            ))
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.optimizer.zero_grad()

    def train_one_epoch(self, train_dataloader):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        epoch = self.training_state.epoch
        config = self.config

        model.train()
        mixup = 0.1
        cri = torch.nn.CrossEntropyLoss().to(device)

        mixup_l = np.random.beta(mixup, mixup)
        step = 0

        for x, y in self.recycle(train_dataloader):
            # Schedule sending to GPU(s)
            x = x.to(device)
            y = y.to(device)

            # Update learning-rate, including stop training if over.
            lr = self.get_lr(step, config)
            if lr is None:
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y_a, y_b = self.mixup_data(x, y, mixup_l)

            # compute output
            logits = model(x)
            c = self.mixup_criterion(cri, logits, y_a, y_b, mixup_l)

            # Accumulate grads
            c.backward()
            step += 1
            self.training_state.global_steps += 1

            if step % config.print_freq == 0:
                c_num = float(c.data.cpu().numpy())
                print(
                    f"[step {step}/{len(train_dataloader)}]: loss={c_num:.5f} (lr={lr:.1e})"
                )

            # Update params
            need_update = step % config.gradient_accumulation_steps == 0
            if need_update:
                optimizer.step()
                optimizer.zero_grad()
            # Sample new mixup ratio for next batch
            mixup_l = np.random.beta(mixup, mixup)

        all_c, all_top1, all_top5 = self.evaluator.evaluate(model, device)

        state = self.training_state
        config = self.config

        state.eval_mAP = float(np.mean(all_top1))
        if state.eval_mAP >= config.target_mAP:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_mAP: {state.eval_mAP}, target_mAP: {config.target_mAP}"
            )
            state.converged_success()

        return state.end_training

    def recycle(self, iterable):
        for i in iterable:
            yield i

    def get_lr(self, step, config):
        """Returns learning-rate for `step` or None at the end."""
        supports = [config.warmup_steps] + config.lr_steps + [config.max_steps]
        # Linear warmup
        if step < supports[0]:
            return config.lr * step / supports[0]
        # End of training
        elif step >= supports[-1]:
            return None
        # Staircase decays by factor of (1.0/lr_gamma)
        else:
            base_lr = config.lr
            for s in supports[1:]:
                if s < step:
                    base_lr *= config.lr_gamma
            return base_lr

    def mixup_data(self, x, y, l):
        """Returns mixed inputs, pairs of targets, and lambda"""
        indices = torch.randperm(x.shape[0]).to(x.device)

        mixed_x = l * x + (1 - l) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b

    def mixup_criterion(self, criterion, pred, y_a, y_b, l):
        return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)
