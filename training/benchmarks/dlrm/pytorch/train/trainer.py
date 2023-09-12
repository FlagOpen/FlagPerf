# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

# std lib
import datetime
import itertools
import time

# 3rd-party lib
import numpy as np
import torch
import torch.utils.data
from torch.types import Device
from apex import parallel

# local lib
from model import create_model
from optimizers import create_mlp_optimizer, create_embedding_optimizer
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState
from driver import Driver, dist_pytorch
from dataloaders.utils import prefetcher
from utils.utils import StepTimer, MetricLogger


class CudaGraphWrapper:

    def __init__(self,
                 model,
                 train_step,
                 parallelize,
                 zero_grad,
                 cuda_graphs=False,
                 warmup_steps=20):

        self.cuda_graphs = cuda_graphs
        self.warmup_iters = warmup_steps
        self.graph = None
        self.stream = None
        self.static_args = None

        self.model = model

        self._parallelize = parallelize
        self._train_step = train_step
        self._zero_grad = zero_grad

        self.loss = None
        self.step = -1

        if cuda_graphs:
            self.stream = torch.cuda.Stream()
        else:
            # if not using graphs, parallelize the model immediately
            # otherwise do this in the warmup phase under the graph stream
            self.model = self._parallelize(self.model)
            self.stream = torch.cuda.default_stream()

    def _copy_input_data(self, *train_step_args):
        if len(train_step_args) != len(self.static_args):
            raise ValueError(
                f'Expected {len(self.static_args)} arguments to train step'
                f'Got: {len(train_step_args)}')

        for data, placeholder in zip(train_step_args, self.static_args):
            if placeholder is None:
                continue
            placeholder.copy_(data)

    def _cuda_graph_capture(self, *train_step_args):
        self._copy_input_data(*train_step_args)
        self.graph = torch.cuda.CUDAGraph()
        self._zero_grad(self.model)
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss = self._train_step(self.model, *self.static_args)
        return self.loss

    def _cuda_graph_replay(self, *train_step_args):
        self._copy_input_data(*train_step_args)
        self.graph.replay()

    def _warmup_step(self, *train_step_args):
        with torch.cuda.stream(self.stream):
            if self.step == 0:
                self.model = self._parallelize(self.model)
                self.static_args = list(train_step_args)
            else:
                self._copy_input_data(*train_step_args)

            self._zero_grad(self.model)
            self.loss = self._train_step(self.model, *self.static_args)
            return self.loss

    def train_step(self, *train_step_args):
        self.step += 1

        if not self.cuda_graphs:
            self._zero_grad(self.model)
            self.loss = self._train_step(self.model, *train_step_args)
            return self.loss

        if self.step == 0:
            self.stream.wait_stream(torch.cuda.current_stream())

        if self.step < self.warmup_iters:
            return self._warmup_step(*train_step_args)

        if self.graph is None:
            torch.cuda.synchronize()
            self._cuda_graph_capture(*train_step_args)

        self._cuda_graph_replay(*train_step_args)
        return self.loss


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 device_mapping, feature_spec):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator
        self.device_mapping = device_mapping
        self.feature_spec = feature_spec

        self.loss_fn = None
        self.mlp_optimizer = None
        self.embedding_optimizer = None
        self.world_size = config.n_device

    def init(self):
        dist_pytorch.main_proc_print("Init progress:")
        self.model = create_model(self.config, self.device,
                                  self.device_mapping, self.feature_spec)
        self.model.to(self.device)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)

        self.mlp_optimizer = create_mlp_optimizer(self.model, self.config)
        self.embedding_optimizer = create_embedding_optimizer(
            self.model, self.config)

        self.lr_scheduler = create_scheduler(self.config, self.mlp_optimizer,
                                             self.embedding_optimizer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.grad_scaler = self.adapter.create_grad_scaler()
        self.timer = StepTimer()

        torch.backends.cudnn.enabled = self.config.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark

    def train_one_epoch(self, train_dataloader):
        model = self.model
        device = self.device
        state = self.training_state
        config = self.config
        evaluator = self.evaluator
        print_freq = config.print_freq
        metric_logger = MetricLogger(delimiter="  ")
        data_stream = torch.cuda.Stream()

        # last one will be dropped in the training loop
        steps_per_epoch = len(train_dataloader) - 1
        test_freq = config.test_freq if config.test_freq is not None else steps_per_epoch - 2

        # Accumulating loss on GPU to avoid memcpyD2H every step
        moving_loss = torch.zeros(1, device=device)

        trainerWrapper = CudaGraphWrapper(model,
                                          self.forward_backward,
                                          self.parallelize,
                                          self.zero_grad,
                                          cuda_graphs=config.cuda_graphs)

        model.train()

        batch_iter = prefetcher(iter(train_dataloader), data_stream)
        max_step = len(train_dataloader)

        for step in range(max_step):
            no_eval_start_time = time.time()
            state.global_steps += 1
            numerical_features, categorical_features, click = next(batch_iter)
            state.num_trained_samples = state.global_steps * \
                dist_pytorch.global_batch_size(self.config)
            self.timer.click(synchronize=(device == 'cuda'))

            pure_compute_start_time = time.time()

            state.global_steps = steps_per_epoch * state.epoch + step

            if step % 10 == 0 or step == max_step - 1:
                print(
                    f"step:{step} global_steps:{state.global_steps} state.num_trained_samples:{state.num_trained_samples}"
                )

            # One of the batches will be smaller because the dataset size
            # isn't necessarily a multiple of the batch size. #TODO isn't dropping here a change of behavior
            if click.shape[0] != config.train_batch_size:
                continue

            self.lr_scheduler.step()
            loss = trainerWrapper.train_step(numerical_features,
                                             categorical_features, click)

            # need to wait for the gradients before the weight update
            torch.cuda.current_stream().wait_stream(trainerWrapper.stream)
            self.weight_update()
            moving_loss += loss
            state.pure_compute_time += time.time() - pure_compute_start_time

            if self.timer.measured is None:
                # first iteration, no step time etc. to print
                continue

            if step == 0:
                print(f"Started epoch {state.epoch}...")
            elif step % config.print_freq == 0:
                # Averaging across a print_freq period to reduce the error.
                # An accurate timing needs synchronize which would slow things down.

                # only check for nan every print_freq steps
                if torch.any(torch.isnan(loss)):
                    print(f'NaN loss encountered. step:{step} loss:{loss}')
                    break

                if state.global_steps < config.benchmark_warmup_steps:
                    metric_logger.update(
                        loss=moving_loss.item() / config.print_freq,
                        lr=self.mlp_optimizer.param_groups[0]["lr"])
                else:
                    metric_logger.update(
                        step_time=self.timer.measured,
                        loss=moving_loss.item() / print_freq,
                        lr=self.mlp_optimizer.param_groups[0]["lr"])

                eta_str = datetime.timedelta(
                    seconds=int(metric_logger.step_time.global_avg *
                                (steps_per_epoch - step)))
                metric_logger.print(
                    header=
                    f"Epoch:[{state.epoch}/{config.max_epoch}] [{step}/{steps_per_epoch}]  eta: {eta_str}"
                )

                moving_loss = 0.

            terminal_condition = state.global_steps % test_freq == 0 and state.global_steps > 0 and state.global_steps / steps_per_epoch >= config.test_after
            if terminal_condition:
                state.no_eval_time += time.time() - no_eval_start_time

                auc, validation_loss = evaluator.evaluate(
                    model, evaluator.eval_dataloader)

                print(
                    f"state.global_steps:{state.global_steps} auc:{state.eval_auc}, validation_loss:{validation_loss}"
                )
                if auc is None:
                    continue
                state.eval_auc = auc

                print(
                    f"Epoch {state.epoch} step {step}. auc {auc:.6f} state.global_steps:{state.global_steps}"
                )
                stop_time = time.time()

                if auc > state.best_auc:
                    state.best_auc = auc
                    state.best_epoch = state.epoch + (
                        (step + 1) / steps_per_epoch)

                if validation_loss < state.best_validation_loss:
                    state.best_validation_loss = validation_loss

                if config.target_auc and state.eval_auc >= config.target_auc:
                    run_time_s = int(stop_time - state.train_time_start_ts)
                    dist_pytorch.main_proc_print(
                        f"AUC:{state.eval_auc} Hit target accuracy AUC {config.target_auc} at epoch "
                        f"{state.global_steps / steps_per_epoch:.2f} in {run_time_s}s. "
                    )
                    state.converged_success()
                    break
                else:
                    print(f"config.target_auc:{config.target_auc}, "
                          f"state.eval_auc:{state.eval_auc}")
            else:
                state.no_eval_time += time.time() - no_eval_start_time

            if (state.epoch >= config.max_epoch) or (
                    config.max_steps
                    and state.global_steps >= config.max_steps):
                print(
                    f"state.global_steps:{state.global_steps} config.max_steps:{config.max_steps} finish training, but not converged."
                )
                state.end_training = True
                break

    def zero_grad(self, model):
        if self.config.Adam_embedding_optimizer or self.config.Adam_MLP_optimizer:
            model.zero_grad()
        else:
            # We don't need to accumulate gradient. Set grad to None is faster than optimizer.zero_grad()
            for param_group in itertools.chain(
                    self.embedding_optimizer.param_groups,
                    self.mlp_optimizer.param_groups):
                for param in param_group['params']:
                    param.grad = None

    def forward_backward(self, model, *args):
        config = self.config
        rank = config.local_rank

        numerical_features, categorical_features, click = args
        world_size = self.world_size
        batch_sizes_per_gpu = [
            config.eval_batch_size // world_size for _ in range(world_size)
        ]
        batch_indices = tuple(
            np.cumsum([0] +
                      list(batch_sizes_per_gpu)))  # todo what does this do

        loss = None
        with torch.cuda.amp.autocast(enabled=self.config.amp):
            output = model(numerical_features, categorical_features,
                           batch_sizes_per_gpu).squeeze()
            loss = self.loss_fn(
                output, click[batch_indices[rank]:batch_indices[rank + 1]])

        self.grad_scaler.scale(loss).backward()

        return loss

    def weight_update(self):
        config = self.config
        world_size = config.n_device
        mlp_optimizer = self.mlp_optimizer

        if not config.freeze_mlps:
            if config.Adam_MLP_optimizer:
                scale_MLP_gradients(mlp_optimizer, world_size)
            self.grad_scaler.step(mlp_optimizer)

        if not config.freeze_embeddings:
            if config.Adam_embedding_optimizer:
                scale_embeddings_gradients(self.embedding_optimizer,
                                           world_size)
            self.grad_scaler.unscale_(self.embedding_optimizer)
            self.embedding_optimizer.step()

        self.grad_scaler.update()

    def parallelize(self, model):
        world_size = self.world_size
        use_gpu = "cpu" not in self.config.base_device.lower()
        if world_size <= 1:
            return model

        if use_gpu:
            model.top_model = parallel.DistributedDataParallel(model.top_model)
        else:  # Use other backend for CPU
            model.top_model = torch.nn.parallel.DistributedDataParallel(
                model.top_model)
        return model


def scale_MLP_gradients(mlp_optimizer: torch.optim.Optimizer, world_size: int):
    for param_group in mlp_optimizer.param_groups[1:]:  # Omitting top MLP
        for param in param_group['params']:
            param.grad.div_(world_size)


def scale_embeddings_gradients(embedding_optimizer: torch.optim.Optimizer,
                               world_size: int):
    for param_group in embedding_optimizer.param_groups:
        for param in param_group['params']:
            if param.grad != None:
                param.grad.div_(world_size)