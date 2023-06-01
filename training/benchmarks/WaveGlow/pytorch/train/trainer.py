import os
import sys
import time

import torch
import torch.utils.data
import numpy as np

from optimizers import create_optimizer
from model import create_model
from loss import create_criterion
from train.evaluator import Evaluator
from train import trainer_adapter
from dataloaders import data_function
from utils.utils import reduce_tensor, save_checkpoint, adjust_learning_rate

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, dist_pytorch
from utils.utils import init_dllogger


class Trainer:

    def __init__(self, training_state, config):
        super(Trainer, self).__init__()
        self.adapter = trainer_adapter
        self.training_state = training_state
        self.config = config
        self.logger = init_dllogger(config)

    def init(self):
        # torch.set_num_threads(1)
        dist_pytorch.main_proc_print("Init progress:")
        self.model = create_model(self.config)
        self.model.to(self.config.device)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model, self.config)
        self.model = self.adapter.model_to_ddp(self.model, self.config)
        self.optimizer = create_optimizer(self.model, self.config)
        self.scaler = self.adapter.create_grad_scaler(self.config)
        self.criterion = create_criterion(self.config)
        self.batch_to_gpu = data_function.get_batch_to_gpu(self.config.name)
        self.world_size = dist_pytorch.get_world_size()
        self.evaluator = Evaluator(self.logger)

    def train_one_epoch(self, epoch, train_loader, val_loader, config,
                        iteration, train_epoch_items_per_sec, val_loss):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()
        # used to calculate avg items/sec over epoch
        reduced_num_items_epoch = 0

        train_epoch_items_per_sec = 0.0
        num_iters = 0
        reduced_loss = 0

        self.model.train()
        if self.world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            torch.cuda.synchronize()
            iter_start_time = time.perf_counter()
            self.logger.log(step=(epoch, i),
                            data={
                                'glob_iter/iters_per_epoch':
                                str(iteration) + "/" + str(len(train_loader))
                            })

            adjust_learning_rate(iteration, epoch, self.optimizer,
                                 self.config.learning_rate,
                                 self.config.anneal_steps,
                                 self.config.anneal_factor,
                                 self.config.local_rank)

            self.model.zero_grad()

            x, y, num_items = self.batch_to_gpu(batch)

            #AMP upstream autocast
            with torch.cuda.amp.autocast(enabled=True):
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

            if self.world_size > 1:
                reduced_loss = reduce_tensor(loss.data, self.world_size).item()
                reduced_num_items = reduce_tensor(num_items.data, 1).item()
            else:
                reduced_loss = loss.item()
                reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            self.logger.log(step=(epoch, i), data={'train_loss': reduced_loss})

            num_iters += 1

            # accumulate number of items processed in this epoch
            reduced_num_items_epoch += reduced_num_items

            if self.config.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               config.grad_clip_thresh)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               config.grad_clip_thresh)
                self.optimizer.step()

            self.model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time
            items_per_sec = reduced_num_items / iter_time
            train_epoch_items_per_sec += items_per_sec

            self.logger.log(step=(epoch, i),
                            data={'train_items_per_sec': items_per_sec})
            self.logger.log(step=(epoch, i),
                            data={'train_iter_time': iter_time})
            iteration += 1

        torch.cuda.synchronize()
        epoch_stop_time = time.perf_counter()
        epoch_time = epoch_stop_time - epoch_start_time

        self.logger.log(step=(epoch, ),
                        data={
                            'train_items_per_sec':
                            (train_epoch_items_per_sec /
                             num_iters if num_iters > 0 else 0.0)
                        })
        self.logger.log(step=(epoch, ), data={'train_loss': reduced_loss})
        self.logger.log(step=(epoch, ), data={'train_epoch_time': epoch_time})

        val_loss, val_items_per_sec = self.evaluator.validate(
            self.model, self.criterion, epoch, iteration, self.world_size, self.world_size > 1, self.batch_to_gpu, self.config.amp, val_loader)

        self.training_state.val_loss = val_loss

        if self.training_state.val_loss <= self.config.target_val_loss:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_val_loss: {self.training_state.val_loss}, target_val_loss: {config.target_val_loss}"
            )
            self.training_state.converged_success()

        if epoch >= config.epochs:
            self.training_state.end_training = True

        if self.config.local_rank == 0:
            self.logger.flush()

        if (config.save_checkpoint and epoch % config.epochs_per_checkpoint
                == 0) and (config.bench_class == ""
                           or config.bench_class == "train"):
            save_checkpoint(self.model, self.optimizer, self.scaler, epoch,
                            self.config.model_config, self.config.output,
                            self.config.name, self.config.local_rank,
                            self.world_size)

        return train_epoch_items_per_sec, val_items_per_sec, val_loss, num_iters
