# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")


import os
import sys
import time

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
from dataloaders.dataset import build_train_dataset, build_eval_dataset
from train.trainer import Trainer
from train.training_state import TrainingState
from train import trainer_adapter
from train.evaluator import Evaluator
from dataloaders.dataloader_wav import build_train_dataloader, build_eval_dataloader
from wav2vec2.logging import init_logger
import numpy as np
from common.utils import print_once
from common import tb_dllogger as logger_init

logger = None


def main():

    global logger
    global config
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)

    dist_pytorch.barrier(config.vendor)

    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    init_helper.set_seed(config.seed + config.local_rank,
                         model_driver.config.vendor)

    train_dataset = build_train_dataset(config.train_subset,
                                        config,
                                        with_labels=False,
                                        training=True)
    valid_dataset = build_eval_dataset(config.valid_subset,
                                       config,
                                       with_labels=False,
                                       training=False)

    world_size = dist_pytorch.get_world_size()
    print_once(f"World size: {world_size}")

    train_dataloader, sampler = build_train_dataloader(
        train_dataset,
        True,
        max_tokens=config.max_tokens,
        max_sentences=config.batch_size,
        max_positions=(config.max_tokens, config.max_tokens),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=config.required_batch_size_multiple,
        seed=config.seed,
        num_shards=world_size,
        shard_id=int(os.getenv("RANK")),
        num_workers=config.num_workers,
        num_concat_batches=config.num_concat_batches)

    eval_dataloader, _ = build_eval_dataloader(
        valid_dataset,
        False,
        max_tokens=config.max_tokens_valid,
        max_sentences=config.batch_size_valid,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=config.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=config.required_batch_size_multiple,
        seed=config.seed,
        num_shards=world_size,
        shard_id=int(os.getenv("RANK", config.local_rank)),
        num_workers=config.num_workers,
        num_concat_batches=config.num_concat_batches)

    evaluator = Evaluator(eval_dataloader)
    training_state = TrainingState()

    train_state = {
        'step': 0,
        'epoch': 1,
        'best_val_loss': float('inf'),
        'best_val_wer': float('inf')
    }

    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config,
                      train_dataloader=train_dataloader,
                      train_state=train_state)

    training_state._trainer = trainer

    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    if not config.do_train:
        return config, training_state

    step, epoch = train_state['step'], train_state['epoch']
    start_step = step
    start_epoch = epoch

    init_logger(config.output_dir, config.ema)

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    init_evaluation_start = time.time()

    print_once('Validating...')
    _, training_state.val_acc, _, = evaluator.validate(
        epoch, step, trainer.model, trainer.criterion, trainer.val_metrics,
        trainer.val_ema_metrics, world_size, config.fp16, config.bf16)

    init_evaluation_end = time.time()

    init_evaluation_info = dict(eval_acc=training_state.val_acc,
                                time=init_evaluation_end -
                                init_evaluation_start)

    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time

    while step < config.max_update and \
                not training_state.end_training: # training loop

        step, epoch, training_state.throughoutputs = trainer.train_all_epoch(
            config, epoch, step, train_dataloader, sampler)

        if 0 < config.epochs_this_job <= epoch + 1 - start_epoch:
            print_once(f'Reached {config.epochs_this_job} epochs in this run.')
            break

        if step >= config.max_update:
            print_once(f'Reached {step} total updates.')
            break

    # finished training
    if step > start_step:
        logger_init.log((), trainer.metrics, scope='train_benchmark')
        logger_init.log((), trainer.val_metrics, scope='val')
        logger_init.log((),
                        trainer.val_ema_metrics,
                        scope='val_ema',
                        flush_log=True)

    print_once(f'Finished after reaching update {step}.')
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time

    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    global_batch_size = dist_pytorch.global_batch_size(config_update)  # TODO
    e2e_time = time.time() - start
    finished_info = {"e2e_time": e2e_time}
    if config_update.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": state.throughoutputs,
            "converged": state.converged,
            "final_loss": state.val_losses,
            "final_acc": state.val_acc,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
