import time
import argparse
import os
import numpy as np
import torch
import random

from dataloaders import (WorkerInitializer, build_train_dataloader,
                         build_eval_dataloaders)
import utils
from train.trainer import Trainer, Evaluator
from train.training_state import TrainingState
# from train.event import TrainingEventCompose, TrainingLogger, BaseTrainingEventInterface

import train
from train.driver import LogEventManager, Driver, Event
from train.driver import mod_util, check, backend, distributed, trainer_adapter

logger = None


def main():
    global logger
    import config

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    driver = Driver()
    driver.setup_config(argparse.ArgumentParser("Glm"))
    driver.setup_modules(train.driver, globals(), locals())
    # mod_util.replace_submodules(train.driver, driver.extern_modules)
    # mod_util.remap_modules(globals(), driver.extern_modules)
    # mod_util.remap_modules(locals(), driver.extern_modules)

    logger = driver.logger

    distributed.init_dist_training_env(config)
    #check.check_config(config)

    utils.check_config(config)

    # events = [
    #     TrainingLogger(config, log_freq=config.log_freq)
    # ]
    # training_event = TrainingEventCompose(interface, events)
    # training_event.launch()

    # global logger
    # logger = events[0].logger

    utils.barrier()
    # training_event.on_init_start()
    driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    worker_seeds, shuffling_seeds = utils.setup_seeds(
        config.seed, config.num_epochs_to_generate_seeds_for, config.device)

    if torch.distributed.is_initialized():
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)

    evaluator = Evaluator(config, None)
    training_state = TrainingState()
    trainer = Trainer(driver=driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)
    training_state._trainer = trainer

    utils.barrier()
    trainer.init()

    eval_dataloader = build_eval_dataloaders(config)

    utils.barrier()
    init_evaluation_start = time.time()
    evaluator.dataloader = eval_dataloader
    score = trainer.evaluator.evaluate(trainer)
    training_state.eval_accuracy = score
    init_evaluation_end = time.time()
    init_evaluation_info = dict(eval_accuracy=score,
                                time=init_evaluation_end -
                                init_evaluation_start)
    # training_event.on_init_evaluate(init_evaluation_info)
    driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    train_dataloader = build_train_dataloader(config, worker_init)

    if not config.do_train:
        return config, training_state

    # training_event.on_init_end()
    driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    utils.barrier()

    epoch = -1
    # training_event.on_train_begin()
    driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time

    while training_state.num_trained_samples < config.max_samples_termination and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        train_dataloader.sampler.set_epoch(epoch)
        trainer.train_one_epoch(train_dataloader)

    # training_event.on_train_end()
    driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":

    now = time.time()
    config, state = main()

    if not utils.is_main_process():
        exit()

    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    if config.do_train:
        training_perf = (utils.global_batch_size(config) *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_accuracy": state.eval_accuracy,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
