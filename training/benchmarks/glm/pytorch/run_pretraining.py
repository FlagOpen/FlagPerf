# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""GLM Pretraining"""

import os
import sys
import time
import torch

import config
from dataloaders import (WorkerInitializer, build_train_dataloader,
                         build_eval_dataloaders)
from train.trainer import Trainer, Evaluator
from train.training_state import TrainingState
from train import trainer_adapter

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Event, dist_pytorch, check
from driver.helper import InitHelper

logger = None


def main():

    from config import mutable_params
    global logger
    global config

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    init_helper = InitHelper(config)
    glm_driver = init_helper.init_driver(globals(), locals())
    logger = glm_driver.logger

    dist_pytorch.init_dist_training_env(config)

    check.check_config(config)

    dist_pytorch.barrier(config.vendor)
    glm_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    worker_seeds, shuffling_seeds = dist_pytorch.setup_seeds(
        config.seed, config.num_epochs_to_generate_seeds_for, config.device)

    if torch.distributed.is_initialized():
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]

    init_helper.set_seed(config.seed, config.vendor)

    worker_init = WorkerInitializer.default(worker_seed)
    train_dataloader = build_train_dataloader(config, worker_init)
    eval_dataloader = build_eval_dataloaders(config)

    evaluator = Evaluator(config, eval_dataloader)
    training_state = TrainingState()
    trainer = Trainer(driver=glm_driver,
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
    score = trainer.evaluator.evaluate(trainer)
    training_state.eval_accuracy = score
    init_evaluation_end = time.time()
    init_evaluation_info = dict(eval_accuracy=score,
                                time=init_evaluation_end -
                                init_evaluation_start)
    glm_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state

    glm_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier(config.vendor)

    glm_driver.event(Event.TRAIN_START)
    raw_train_start_time = time.time()

    epoch = 0
    while training_state.num_trained_samples < config.max_samples_termination and not training_state.end_training:
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
        epoch += 1

    glm_driver.event(Event.TRAIN_END)
    training_state.raw_train_time = time.time() - raw_train_start_time

    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config_upadted, state = main()

    if not dist_pytorch.is_main_process():
        sys.exit()

    e2e_time = time.time() - now
    if config_upadted.do_train:
        training_perf = (dist_pytorch.global_batch_size(config_upadted) *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "global_steps": state.global_steps,
            "num_trained_samples": state.num_trained_samples,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_accuracy": state.eval_accuracy,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
            "pure_training_computing_time": state.pure_compute_time,
            "throughput(ips)_raw": state.num_trained_samples / state.raw_train_time,
            "throughput(ips)_no_eval": state.num_trained_samples / state.no_eval_time,
            "throughput(ips)_pure_compute": state.num_trained_samples / state.pure_compute_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
