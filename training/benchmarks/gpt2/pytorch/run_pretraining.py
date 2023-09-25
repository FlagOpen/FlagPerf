# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""GPT2 Pretraining"""

import argparse
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch

from train.trainer import Trainer
from train import trainer_adapter
from train.evaluator import Evaluator
from train.training_state import TrainingState
from dataloaders.gpt_dataset import build_train_test_datasets, build_train_test_data_dataloaders

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch, check

logger = None


def main():
    import config
    global logger

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    gpt2_driver = Driver(config, config.mutable_params)
    gpt2_driver.setup_config(argparse.ArgumentParser("GPT2"))
    gpt2_driver.setup_modules(globals(), locals())

    logger = gpt2_driver.logger
    dist_pytorch.init_dist_training_env(config)

    check.check_config(config)

    dist_pytorch.barrier(config.vendor)
    gpt2_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.global_batch_size = config.train_batch_size * config.n_device * config.gradient_accumulation_steps

    train_data_path = os.path.join(config.data_dir, config.train_data_prefix)
    test_data_path = os.path.join(config.data_dir, config.test_data_prefix)
    build_train_test_dataset_fn = partial(
        build_train_test_datasets,
        seq_length=config.seq_length,
        seed=config.seed,
        skip_warmup=(not config.mmap_warmup),
        train_data_prefix=train_data_path,
        test_data_prefix=test_data_path,
    )
    train_dataloader, eval_dataloader= build_train_test_data_dataloaders(build_train_test_dataset_fn)
    
    evaluator = Evaluator(config, eval_dataloader)
    training_state = TrainingState()
    trainer = Trainer(driver=gpt2_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)

    training_state._trainer = trainer

    dist_pytorch.barrier(config.vendor)
    trainer.init()

    dist_pytorch.barrier(config.vendor)
    init_evaluation_start = time.time()
    training_state.eval_lambada_acc = evaluator.evaluate(
        trainer)
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_lambada_acc=training_state.eval_lambada_acc,
        time=init_evaluation_end - init_evaluation_start)
    gpt2_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state

    gpt2_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier(config.vendor)
    epoch = -1
    gpt2_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
    gpt2_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config_updated, state = main()

    if not dist_pytorch.is_main_process():
        exit()

    e2e_time = time.time() - now
    trained_samples = state.num_trained_samples
    if config_updated.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "train_samples": trained_samples,
            "train_time": state.raw_train_time,
            "train_no_eval_time": state.no_eval_time,
            "pure_training_computing_time": state.pure_compute_time,
            "throughput(sps)_raw":
            trained_samples / state.raw_train_time,
            "throughput(sps)_no_eval":
            trained_samples / state.no_eval_time,
            "throughput(sps)_pure_compute":
            trained_samples / state.pure_compute_time,
            "converged": state.converged,
            "final_accuracy": state.eval_lambada_acc,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

