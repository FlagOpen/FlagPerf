# coding=utf-8
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT Pretraining"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import time
from ast import parse
from copy import copy
from inspect import isclass
from operator import mod

import numpy as np
import paddle


import train
from dataloaders import WorkerInitializer
from dataloaders.dataloader import PretrainingDataloaders
from train.driver import (Driver, Event, check, distributed, mod_util,
                          trainer_adapter)
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

logger = None


def main():
    global logger
    import config

   
    if config.use_env and 'PADDLE_TRAINER_ID' in os.environ:
        config.local_rank = int(os.environ['PADDLE_TRAINER_ID'])

    driver = Driver()
    driver.setup_config(argparse.ArgumentParser("Bert"))
    driver.setup_modules(train.driver, globals(), locals())


    logger = driver.logger

    distributed.init_dist_training_env(config)


    check.check_config(config)

    distributed.barrier()
    driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time
    worker_seeds, shuffling_seeds = distributed.setup_seeds(
        config.seed, config.num_epochs_to_generate_seeds_for, config.device)

    worker_seed = worker_seeds[0]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)
    gen = paddle.seed(worker_seed)
    gen.manual_seed(worker_seed)

    evaluator = Evaluator(
        config.eval_dir,
        global_batch_size=distributed.global_batch_size(config),
        max_steps=config.max_steps,
        worker_init=worker_init,
    )
    grad_scaler = trainer_adapter.create_grad_scaler()

    training_state = TrainingState()
    trainer = Trainer(driver, trainer_adapter, evaluator,
                      training_state)
    training_state._trainer = trainer

    distributed.barrier()
    trainer.init()
    distributed.barrier()
    init_evaluation_start = time.time()
    eval_loss, eval_mlm_acc = evaluator.evaluate(trainer)
    training_state.eval_loss = eval_loss
    training_state.eval_mlm_accuracy = eval_mlm_acc
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_loss=eval_loss,
        eval_mlm_accuracy=eval_mlm_acc,
        time=init_evaluation_end - init_evaluation_start
    )
    driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state


    dataloader = PretrainingDataloaders(
        config.train_dir,
        max_predictions_per_seq=config.max_predictions_per_seq,
        batch_size=config.train_batch_size,
        seed=shuffling_seeds, num_files_per_iter=1,
        worker_init=worker_init, 
    )

    driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    distributed.barrier()

    epoch = -1
    driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        dataloader.set_epoch(epoch)
        trainer.train_one_epoch(dataloader)
    driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (
        raw_train_end_time - raw_train_start_time) / 1e+3
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config, state = main()

    if not distributed.is_main_process():
        exit()

    gpu_count = config.n_device
    e2e_time = time.time() - now
    if config.do_train:
        training_perf = (distributed.global_batch_size(config)
                         * state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_loss,
            "final_mlm_accuracy": state.eval_mlm_accuracy,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
