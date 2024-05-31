# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import os
import sys
import argparse
from typing import Dict

import torch
import torch.utils.data
from torch.types import Device
from torch.cuda.amp import autocast

from model import create_model
from model.model import MoFlow, MoFlowLoss
from model.utils import initialize
from train.evaluator import Evaluator
from train.training_state import TrainingState
from runtime.generate import infer
from runtime.distributed_utils import reduce_tensor
from runtime.common import save_state
from misc.utils import check_validity, convert_predictions_to_mols
from misc.config import CONFIGS, Config
# add benchmarks directory to sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, dist_pytorch


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, args,
                 perf_logger, acc_logger, train_dataloader):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.evaluator = evaluator
        self.args = args
        self.perf_logger = perf_logger
        self.acc_logger = acc_logger
        self.train_dataloader = train_dataloader

        self.scaler = None
        self.config = None
        self.clip_grad = None
        self.loss_module = None
        self.model_callable = None
        self.loss_callable = None
        self.model = None

    def init(self):
        args = self.args
        dist_pytorch.main_proc_print("Init progress:")
        self.config = CONFIGS[self.args.dataset_name]
        self.model = create_model(self.config)
        self.model.to(self.device)
        device = args.device
        x, adj, *_ = next(iter(self.train_dataloader))
        x = x.to(device)
        adj = adj.to(device)
        with autocast(enabled=args.amp):
            initialize(self.model, (adj, x))

        self.model.to(memory_format=torch.channels_last)
        adj.to(memory_format=torch.channels_last)

        if args.jit:
            self.model.bond_model = torch.jit.script(self.model.bond_model)
            self.model.atom_model = torch.jit.script(self.model.atom_model)

        # make one pass in both directions to make sure that model works
        with torch.no_grad():
            _ = self.model(adj, x)
            _ = self.model.reverse(
                torch.randn(args.train_batch_size,
                            self.config.z_dim,
                            device=device))

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model, self.args)
        self.model = self.adapter.model_to_ddp(self.model, self.args)
        self.loss_module = MoFlowLoss(self.config)
        self.loss_module.to(self.device)
        self.loss_module = self.adapter.model_to_ddp(self.loss_module,
                                                     self.args)
        self.model_callable, self.loss_callable = self._get_callables()

        self.optimizer = self.adapter.create_optimizer(self.model, self.args,
                                                       self.loss_module)
        self.scaler = self.adapter.create_grad_scaler(self.args)
        self.clip_grad = self.adapter.create_clip_grad()

    def _get_callables(self):
        args = self.args
        is_distributed = args.distributed
        model_callable, loss_callable = None, None
        if is_distributed:
            model_callable = self.model.module
            loss_callable = self.loss_module.module
        else:
            model_callable = self.model
            loss_callable = self.loss_module
        return model_callable, loss_callable

    def train_one_epoch(self, train_dataloader, step):
        model = self.model
        args = self.args
        device = self.device
        epoch = self.training_state.epoch
        clip_grad_norm_ = self.clip_grad
        world_size = args.n_device
        local_rank = self.args.local_rank
        is_distributed = args.distributed

        if step > 0 and step > self.training_state.global_steps:
            self.training_state.global_steps = step

        if local_rank == 0:
            self.acc_logger.reset()

        print(f"Epoch: {epoch}")
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        noeval_start_time = time.time()

        for i, batch in enumerate(train_dataloader):
            if local_rank == 0:
                self.perf_logger.update()
            self.training_state.global_steps += 1
            self.optimizer.zero_grad()
            x = batch[0].to(device)
            adj = batch[1].to(device=device, memory_format=torch.channels_last)

            pure_compute_start_time = time.time()

            # Forward, backward and optimize
            with_cuda_graph = (
                args.cuda_graph
                and self.training_state.global_steps >= args.warmup_steps
                and x.size(0) == args.train_batch_size)

            with autocast(enabled=args.amp, cache_enabled=not with_cuda_graph):
                output = model(adj, x, with_cuda_graph=with_cuda_graph)
                nll_x, nll_adj = self.loss_module(*output)
                loss = nll_x + nll_adj

            if args.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(model.parameters(), args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), args.clip)
                self.optimizer.step()

            self.training_state.pure_compute_time += time.time(
            ) - pure_compute_start_time

            # Print log info
            if (i + 1) % args.log_interval == 0:
                nll_x_value = reduce_tensor(nll_x, world_size).item()
                nll_adj_value = reduce_tensor(nll_adj, world_size).item()
                loss_value = nll_x_value + nll_adj_value

                if local_rank == 0:
                    self.acc_logger.update({
                        'loglik': loss_value,
                        'nll_x': nll_x_value,
                        'nll_adj': nll_adj_value
                    })

                    self.acc_logger.summarize(step=(epoch, i, i))
                    self.perf_logger.summarize(step=(epoch, i, i))

            if self.training_state.global_steps >= args.steps:
                break

        self.training_state.num_trained_samples += len(
            train_dataloader.dataset)
        self.training_state.no_eval_time += time.time() - noeval_start_time

        if (epoch + 1) % args.eval_epochs == 0:
            with autocast(enabled=args.amp):
                metrics = run_validation(self.model, self.config,
                                         self.loss_callable.ln_var.item(),
                                         args, is_distributed, world_size,
                                         device)
                dist_pytorch.main_proc_print(
                    f"epoch:{epoch+1}, metrics:{metrics}")

            if local_rank == 0:
                self.acc_logger.update(metrics)

        # The same report for each epoch
        if local_rank == 0:
            self.acc_logger.summarize(step=(epoch, ))
            self.perf_logger.summarize(step=(epoch, ))

        # Save the model checkpoints
        if (epoch + 1) % args.save_epochs == 0:
            if local_rank == 0 or not is_distributed:
                save_state(args.results_dir,
                           self.model_callable,
                           self.optimizer,
                           self.loss_callable.ln_var.item(),
                           epoch,
                           keep=3)


def run_validation(model: MoFlow, config: Config, ln_var: float,
                   args: argparse.Namespace, is_distributed: bool,
                   world_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    model_callable = model.module if is_distributed else model

    result = infer(model_callable,
                   config,
                   device=device,
                   ln_var=ln_var,
                   batch_size=args.eval_batch_size,
                   temp=args.temperature)
    mols = convert_predictions_to_mols(*result,
                                       correct_validity=args.correct_validity)
    validity_info = check_validity(mols)
    valid_ratio = torch.tensor(validity_info['valid_ratio'],
                               dtype=torch.float32,
                               device=device)
    unique_ratio = torch.tensor(validity_info['unique_ratio'],
                                dtype=torch.float32,
                                device=device)
    valid_value = reduce_tensor(valid_ratio, world_size).detach().cpu().numpy()
    unique_value = reduce_tensor(unique_ratio,
                                 world_size).detach().cpu().numpy()
    model.train()
    return {'valid': valid_value, 'unique': unique_value}
