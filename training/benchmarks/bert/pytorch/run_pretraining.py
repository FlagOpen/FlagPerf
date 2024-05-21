"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import utils

from dataloaders import WorkerInitializer
from dataloaders.dataloader import PretrainingDataloaders
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from train import trainer_adapter

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
import driver
from driver import Driver, Event, dist_pytorch, check

logger = None


def main():
    import config
    from config import mutable_params
    global logger

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    bert_driver = Driver(config, config.mutable_params)
    bert_driver.setup_config(argparse.ArgumentParser("Bert"))
    bert_driver.setup_modules(driver, globals(), locals())
    config.distributed = dist_pytorch.get_world_size() > 1

    logger = bert_driver.logger

    world_size = int(os.getenv('WORLD_SIZE', "1"))
    if world_size == 1:
        config.local_rank = -1
    dist_pytorch.init_dist_training_env(config)
    utils.check_config(config)

    dist_pytorch.barrier(config.vendor)
    if world_size == 1:
        config.local_rank = 0

    bert_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    worker_seeds, shuffling_seeds = dist_pytorch.setup_seeds(
        config.seed, config.num_epochs_to_generate_seeds_for, config.device)

    if torch.distributed.is_initialized():
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)
    pool = ProcessPoolExecutor(1)
    evaluator = Evaluator(
        config.eval_dir,
        proc_pool=pool,
        global_batch_size=dist_pytorch.global_batch_size(config),
        max_steps=config.max_steps,
        worker_init=worker_init,
        use_cache=config.cache_eval_data)
    training_state = TrainingState()
    trainer = Trainer(driver=bert_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      grad_scaler=config.grad_scaler,
                      device=config.device)
    training_state._trainer = trainer

    dist_pytorch.barrier(config.vendor)
    trainer.init()

    dist_pytorch.barrier(config.vendor)
    init_evaluation_start = time.time()
    eval_loss, eval_mlm_acc = evaluator.evaluate(trainer)
    training_state.eval_loss = eval_loss
    training_state.eval_mlm_accuracy = eval_mlm_acc
    init_evaluation_end = time.time()
    init_evaluation_info = dict(eval_loss=eval_loss,
                                eval_mlm_accuracy=eval_mlm_acc,
                                time=init_evaluation_end -
                                init_evaluation_start)
    bert_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
    if not config.do_train:
        return config, training_state

    dataloader = PretrainingDataloaders(
        config.train_dir,
        max_predictions_per_seq=config.max_predictions_per_seq,
        batch_size=config.train_batch_size,
        seed=shuffling_seeds,
        num_files_per_iter=1,
        worker_init=worker_init,
        pool=pool,
    )

    bert_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3
    epoch = -1
    bert_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        dataloader.set_epoch(epoch)
        trainer.train_one_epoch(dataloader)
    bert_driver.event(Event.TRAIN_END)

    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config, state = main()

    if not dist_pytorch.is_main_process():
        exit()

    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    trained_samples = dist_pytorch.get_world_size() * state.num_trained_samples
    if config.do_train:

        finished_info = {
            "e2e_time": e2e_time,
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
            "final_mlm_accuracy": state.eval_mlm_accuracy,
        }
    else:
        finished_info = {"e2e_time": e2e_time}

    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

    if np.isnan(float(state.eval_loss)):
        raise Exception("Error: state.eval_loss find NaN, please check!")
