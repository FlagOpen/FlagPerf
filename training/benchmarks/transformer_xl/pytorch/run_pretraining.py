# Copyright (c) 2023 BAAI. All rights reserved.
#
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

from __future__ import absolute_import, division, print_function

import os
import sys

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

import argparse
import os
import random
import sys
import time
from contextlib import contextmanager

import driver
import numpy as np
import torch
import utils
from dataset import WorkerInitializer, create_dataset
from driver import Driver, Event, check, dist_pytorch
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

logger = None


@contextmanager
def event_guard(driver, start, end):
    driver.event(start)
    yield
    driver.event(end)


def main():
    import config
    from config import mutable_params

    global logger

    if config.use_env and "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])

    instance_driver = Driver(config, mutable_params)
    instance_driver.setup_config(argparse.ArgumentParser("Transformer XL"))
    instance_driver.setup_modules(driver, globals(), locals())
    config.distributed = dist_pytorch.get_world_size() > 1

    logger = instance_driver.logger

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        config.local_rank = -1
    dist_pytorch.init_dist_training_env(config)
    utils.check_config(config)

    dist_pytorch.barrier(config.vendor)
    if world_size == 1:
        config.local_rank = 0

    with event_guard(instance_driver, Event.INIT_START, Event.INIT_END):
        init_start_time = logger.previous_log_time

        worker_seeds, shuffling_seeds = dist_pytorch.setup_seeds(
            config.seed, config.num_epochs_to_generate_seeds_for, config.device
        )

        if torch.distributed.is_initialized():
            worker_seed = worker_seeds[torch.distributed.get_rank()]
        else:
            worker_seed = worker_seeds[0]

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        worker_init = WorkerInitializer.default(worker_seed)
        train_dataset, eval_dataset = create_dataset(config)
        training_state = TrainingState()
        evaluator = Evaluator(config, eval_dataset)
        trainer = Trainer(
            driver=instance_driver,
            evaluator=evaluator,
            state=training_state,
            device=config.device,
        )
        training_state._trainer = trainer

        dist_pytorch.barrier(config.vendor)
        trainer.init(config)

        dist_pytorch.barrier(config.vendor)
        init_evaluation_start = time.time()
        eval_loss, eval_ppl = evaluator.evaluate(trainer)
        training_state.eval_loss = eval_loss
        init_evaluation_end = time.time()
        init_evaluation_info = dict(
            eval_loss=eval_loss, eval_ppl=eval_ppl, time=init_evaluation_end - init_evaluation_start
        )
        instance_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e3
    epoch = -1

    with event_guard(instance_driver, Event.TRAIN_START, Event.TRAIN_END):
        raw_train_start_time = logger.previous_log_time
        while (
            config.max_steps is None or training_state.global_step < config.max_steps
        ) and not training_state.end_training:
            epoch += 1
            training_state.epoch = epoch
            trainer.train_one_epoch(train_dataset)
    raw_train_end_time = logger.previous_log_time
    training_state.traintime = (raw_train_end_time - raw_train_start_time) / 1e3
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config, state = main()

    if not dist_pytorch.is_main_process():
        exit()

    device_count = config.n_device
    e2e_time = time.time() - now
    trained_samples = dist_pytorch.get_world_size() * state.num_trained_samples
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "train_time": state.traintime,
            "train_no_eval_time": state.noevaltime,
            "pure_training_computing_time": state.purecomputetime,
            "throughput(sps)_raw": trained_samples / state.traintime,
            "throughput(sps)_no_eval": trained_samples / state.noevaltime,
            "throughput(sps)_pure_compute": trained_samples / state.purecomputetime,
            "converged": state.converged,
            "final_ppl_accuracy": state.ppl,
        }
    else:
        finished_info = {"e2e_time": e2e_time}

    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

    if np.isnan(float(state.eval_loss)):
        raise Exception("Error: state.eval_loss find NaN, please check!")
