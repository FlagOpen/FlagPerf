"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from copy import copy
import os
import time

import numpy as np
import torch
import random

import utils
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from config.config_manager import print_config
from dataloaders.tokenization_gpt2 import GPT2Tokenizer
from dataloaders.dataloader import load_data
# from train.event import TrainingEventCompose, TrainingLogger, BaseTrainingEventInterface

from train.driver import mod_util, check, backend, distributed, trainer_adapter
from train.driver import Driver, Event
import train

logger = None

def main():
    import config
    global logger

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    driver = Driver()
    driver.setup_config(argparse.ArgumentParser("CPM"))
    driver.setup_modules(train.driver, globals(), locals())

    logger = driver.logger

    distributed.init_dist_training_env(config)


    # parser = argparse.ArgumentParser("CPM")
    # config.activate_config_env(parser=parser, with_config_env_name=True)

    # interface: BaseTrainingEventInterface = config.training_event(config)
    # config.training_event_instance = interface

    # device, num_gpus = interface.init_distributed_environment()
    # config.device = device
    # config.n_gpu = num_gpus

    utils.check_config(config)

    # events = [
    #     TrainingLogger(config, log_freq=config.log_freq)
    # ]
    # training_event = TrainingEventCompose(interface, events)
    # training_event.launch()

    # global logger
    # logger = events[0].logger

    utils.barrier()
    driver.event(Event.INIT_START)
    # training_event.on_init_start()
    init_start_time = logger.previous_log_time

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # get the tokenizer
    base_path = os.path.abspath(os.path.dirname(__file__))
    tokenizer = GPT2Tokenizer(os.path.join(base_path, 'dataloaders', config.tokenizer_path, 'vocab.json'), os.path.join(base_path, 'dataloaders', config.tokenizer_path, 'chinese_vocab.model'))
    train_dataloader, _ = load_data(config, 'train', tokenizer, 1)
    eval_dataloader, _ = load_data(config, 'valid', tokenizer, 1)
    print(f"train_dataset length:{len(train_dataloader.dataset)}")
    print(f"train length:{len(train_dataloader)}")
    print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
    print(f"eval length:{len(eval_dataloader)}")

    evaluator = Evaluator(config, eval_dataloader)
    training_state = TrainingState()
    # trainer = Trainer(config, training_event, evaluator, training_state, device=device)
    trainer = Trainer(driver=driver, adapter=trainer_adapter, evaluator=evaluator,
                      training_state=training_state, device=config.device, config=config)


    training_state._trainer = trainer

    utils.barrier()
    trainer.init()

    utils.barrier()
    init_evaluation_start = time.time()
    training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(trainer)
    init_evaluation_end = time.time()
    init_evaluation_info = dict(eval_loss=training_state.eval_avg_loss,
                eval_embedding_average=training_state.eval_embedding_average,
                time=init_evaluation_end - init_evaluation_start)
    # training_event.on_init_evaluate(init_evaluation_info)
    driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state,  init_evaluation_info["time"]

    # training_event.on_init_end()
    driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    utils.barrier()
    epoch = -1
    # training_event.on_train_begin()
    driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time    
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
    # training_event.on_train_end()
    driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time    
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e+3    
    return config, training_state

if __name__ == "__main__":
    now = time.time()
    res = main()
    print('main out out length is', len(res))
    config, state = res

    if not utils.is_main_process():
        exit()

    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    training_perf = (utils.global_batch_size(config) * state.global_steps) / state.raw_train_time
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_avg_loss,
            "final_mlm_accuracy": state.eval_embedding_average,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
