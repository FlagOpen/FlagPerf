# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.types import Device
import math
import time
import collections

from train.evaluator import Evaluator
from train.training_state import TrainingState

from driver import Driver, Event
from fairseq.ddp_trainer import DDPTrainer
from fairseq.models import build_model
from fairseq.data import data_utils
from fairseq import distributed_utils


class Trainer(DDPTrainer):

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = build_model(config).to(device)
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.ddp_trainer = None
        super(Trainer, self).__init__(self.config, self.model)

    def init(self, train_dataloader):
        load_checkpoint(self.config, self, train_dataloader)
        # Send a dummy batch to warm the caching allocator
        src_dict, tgt_dict = data_utils.load_dictionaries(self.config)
        add_extra_items_to_checkpoint({'src_dict': src_dict, 'tgt_dict': tgt_dict})
        dummy_batch = data_utils.get_dummy_batch(self.config.max_tokens, src_dict, tgt_dict)
        self.dummy_train_step(dummy_batch)


    def train_one_epoch(self, train_dataloader, valid_dataloader):
        """Train the model for one epoch."""
        args = self.config
        epoch_itr = train_dataloader
        trainer = self
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)
        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr()

        # update parameters every N batches
        if epoch_itr.epoch <= len(args.update_freq):
            update_freq = args.update_freq[epoch_itr.epoch - 1]
        else:
            update_freq = args.update_freq[-1]

        num_batches = len(epoch_itr)

        no_eval_start_time = time.time()

        trainer.get_throughput_meter().reset()
        for i, sample in enumerate(itr):
            state.global_steps += 1
            update_params = not (i < num_batches - 1 and (i + 1) % update_freq > 0)
            if update_params:
                driver.event(Event.STEP_BEGIN, step=state.global_steps)
            trainer.train_step(sample, update_params=update_params, last_step=(i == len(itr)-1))
            if not update_params:
                continue
            state.lr = trainer.get_lr()
            state.loss = self.avg_loss_meter.avg
            other_state = {"tokens/s": self.throughput_meter.avg}
            driver.event(Event.STEP_END,
                         message=state.to_dict(**other_state),
                         step=state.global_steps,
                         loss=state.loss)
            # ignore the first mini-batch in words-per-second calculation
            if i == 0:
                trainer.get_throughput_meter().reset()

            end_training = self.detect_training_status(state)
            if end_training:
                break

        state.total_tokens += self.throughput_meter.n
        state.no_eval_time += time.time() - no_eval_start_time
        if epoch_itr.epoch % args.validate_interval == 0:
            state.valid_loss = self.validate(valid_dataloader)
            eval_start = time.time()
            state.test_bleu = self.evaluator.evaluate(trainer)
            eval_end = time.time()
            eval_result = dict(global_steps=state.global_steps,
                               valid_loss=state.valid_loss,
                               eval_acc=state.test_bleu,
                               time=eval_end - eval_start)
            driver.event(Event.EVALUATE, eval_result)
            if state.test_bleu >= args.target_bleu:
                state.converged_success()

        trainer.lr_step(epoch_itr.epoch, state.valid_loss)
        save_checkpoint(args, trainer, epoch_itr, state.valid_loss)
        torch.cuda.synchronize()
        driver.event(Event.EPOCH_END, state.epoch)

    def detect_training_status(self, state):
        config = self.config
        max_update = config.max_update or math.inf

        if state.end_training \
                or (self.get_num_updates() >= max_update) \
                or (not state.lr >= config.min_lr):
            state.end_training = True

        return state.end_training

    def validate(self, dataloader):
        itr = dataloader.next_epoch_itr(shuffle=False)
        subset_losses = []
        for sample in itr:
            loss = self.valid_step(sample)
            subset_losses.append(loss)
        loss = sum(subset_losses) / (len(subset_losses) or 1)
        return loss


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if epoch_itr.epoch % args.save_interval != 0:
        return
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = end_of_epoch and not args.no_epoch_checkpoints
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn)
                   for fn, cond in checkpoint_conds.items() if cond]
    if checkpoints:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)


def add_extra_items_to_checkpoint(items):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(items)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
