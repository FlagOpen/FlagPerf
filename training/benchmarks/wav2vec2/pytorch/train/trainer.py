import torch
from torch.types import Device
import os
import sys
import time
from dataloaders.dataset import build_train_dataset, build_eval_dataset

from model import create_model
from schedulers import create_scheduler

from train.evaluator import Evaluator
from train.training_state import TrainingState

import config

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch

#cus
from itertools import cycle, islice
from contextlib import suppress as empty_context
from wav2vec2.criterion import Wav2vecCriterion
from common.fairseq.utils import multiply_grads
from common.helpers import (to_gpu, apply_multi_tensor_ema)
from common.optimizers import lr_poly_policy
from wav2vec2.logging import W2v2Metrics
from common import tb_dllogger as logger
from functools import partial


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
        self.overflow_buf = None

    def init(self):
        self.model = create_model(config)
        self.model = self._init_model(self.model, self.device)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)
        print(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.optim = self.optimizer
        self.grad_scaler = self.adapter.create_grad_scaler()



        Metrics = W2v2Metrics
        self.criterion = Wav2vecCriterion(config)
        kw = {'benchmark_epochs': config.benchmark_epochs_num, 'cuda': not config.cpu}
        self.metrics = Metrics(**kw)

        lr_kw = {'initial_lr_scale': config.initial_lr_scale,
                'final_lr_scale': config.final_lr_scale,
                'warmup_steps': config.warmup_updates,
                'hold_steps': config.hold_updates,
                'num_steps': config.max_update,
                'lr': config.lr}
        if config.lr_policy == 'poly':
            self.adjust_lr = partial(lr_poly_policy, power=config.lr_poly_power, **lr_kw)
        else:
            raise ValueError
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.adjust_lr(1, self.optim)

        self.val_metrics = Metrics(scopes=['val'], **kw)
        self.val_ema_metrics = Metrics(scopes=['val_ema'], **kw)


    def _init_model(self, model, device):
        model = model.to(device)
        return model

    def train_all_epoch(self, config, epoch, step, train_dataloader, 
                        sampler,checkpointer,train_state):
        
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        self.metrics.start_epoch(epoch)
        sampler.set_epoch(epoch)
        self.optim.zero_grad()

        itr = islice(train_dataloader, self.steps_per_epoch * config.update_freq)
        self.model.train()
        step_start_time = time.time()
        other_state = dict()
        
        for batch, accum_batches in zip(itr, cycle(range(config.update_freq))):

            state.global_steps += 1

            # driver.event(Event.STEP_BEGIN, step=state.global_steps)
            
            if accum_batches == 0:
                step += 1
                self.model.set_num_updates(step)
                self.metrics.start_iter(accum_batches)
            to_gpu(batch, fp16=config.fp16, bf16=config.bf16)
            self.model.cuda()
            world_size = dist_pytorch.get_world_size()
            multi_gpu = world_size > 1

            # use context manager to prevent redundant sync of gradients
            if (multi_gpu and accum_batches + 1 < config.update_freq):
                ctx = self.model.no_sync()
            else:
                ctx = empty_context()
            with ctx:
                loss, _, logging_output = self.criterion(self.model, batch)
                if config.fp16 or config.bf16:
                    self.optim.backward(loss)
                else:
                    self.scaler.scale(loss).backward()

            self.metrics.log_scalars(logging_output)

            if (accum_batches + 1) % config.update_freq == 0:
                self.metrics.all_reduce(world_size)

                # scales gradients update by world_size
                # (to restore sum of gradients - see (*))
                # divided by step_ntoks to average over tokens.
                grads_mult_factor = world_size / self.metrics.partials['sample_size']

                if config.optimizer == 'adam' and not (config.fp16 or config.bf16):
                    # adam and non-amp optimizer - can use 'scale' kwarg for step
                    # and defer grad multiplication
                    pass
                elif config.fp16 or config.bf16:
                    self.optim.multiply_grads(grads_mult_factor)
                else:
                    multiply_grads(self.optim, grads_mult_factor)

                try:
                    if config.fp16 or config.bf16:
                        # calculate grad norm, maybe clip
                        grad_norm = self.optim.clip_grad_norm(config.clip_norm)

                    if config.optimizer == 'adam' and not (config.fp16 or config.bf16):
                        self.scaler.step(self.optim, scale=1. / grads_mult_factor)
                    else:
                        self.scaler.step(self.optim)

                    self.scaler.update()
                    self.model.set_num_updates(step)

                except OverflowError as e:
                    # print_once(f"Grad overflow, ignoring grad. {str(e)}")
                    grad_norm = torch.tensor(0.0).cuda()

                self.optim.zero_grad()

                if config.ema > 0.0:
                    apply_multi_tensor_ema(config.ema, *mt_ema_params)

                if config.fp16 or config.bf16:
                    self.metrics['loss_scale'] = self.optim.scaler.loss_scale

                self.metrics['lr'] = self.optim.param_groups[0]['lr']
                self.metrics.accumulate()
                self.metrics.finish_iter()

                if step % config.log_frequency == 0:
                    self.metrics.finish_logging_interval()
                    epoch_step = step % self.steps_per_epoch or self.steps_per_epoch
                    logger.log((epoch, epoch_step, self.steps_per_epoch),
                               self.metrics, scope='train', tb_iter=step)
                self.adjust_lr(step, self.optim)
            
                if self.can_do_eval(state):
                    eval_start = time.time()
                    print('Validating...')
                    val_losses, val_acc, val_wer = self.evaluator.validate(
                                epoch, step, self.evaluator.dataloader, self.model, self.criterion,
                                self.val_metrics, self.val_ema_metrics, world_size, config.fp16, config.bf16)
                    eval_end = time.time()
                    
                    checkpointer.maybe_save(self.model, None, self.optim, self.scaler, train_state,
                                            step, epoch, val_losses, val_wer, config)
                    eval_result = None


                    state.val_losses, state.val_acc = val_losses[0], val_acc,

                    end_training = self.detect_training_status(state)

                    eval_result = dict(global_steps=state.global_steps,
                                        eval_loss=state.val_losses,
                                        eval_acc=state.val_acc,
                                        time=eval_end - eval_start)
                    if eval_result is not None:
                            driver.event(Event.EVALUATE, eval_result)
                    
                    if end_training:
                        break
          
            assert step <= self.steps_per_epoch * epoch
            if step >= config.max_update:
                break
            
            step_end_time = time.time()
            step_total_time = step_end_time - step_start_time
            step_start_time = step_end_time

            # NOTE this will brake when resuming training on a different dataset
            # end of iter
        driver.event(Event.EPOCH_END, state.epoch)
        self.metrics.finish_epoch()

        epoch += 1  # end of epoch
        train_throughoutput = self.metrics.get_metrics("train", 'dll')['train_ntokens/s']
        return step, epoch, train_throughoutput

    def detect_training_status(self, state):
        config = self.config
        if state.val_acc >= config.target_acc:
            state.converged_success()

        return state.end_training

    def init_train_config(self, config, train_dataloader):
        assert config.update_freq >= 1

        if config.ema > 0.0:
            raise NotImplementedError(
                "EMA disabled, see https://github.com/pytorch/pytorch/issues/28594"
            )
        else:
            ema_model = None

        self.steps_per_epoch = len(train_dataloader) // config.update_freq

        return ema_model

    def can_do_eval(self, state):
        do_eval = all([
            state.global_steps % 100 == 0,
            state.global_steps > 1,
        ])
        return do_eval