"""CPM Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import random

import numpy as np
import torch

from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.tokenization_gpt2 import GPT2Tokenizer
from dataloaders.dataloader import load_data
import train
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

    cpm_driver = Driver(config, config.mutable_params)
    cpm_driver.setup_config(argparse.ArgumentParser("CPM"))
    cpm_driver.setup_modules(driver, globals(), locals())

    logger = cpm_driver.logger
    dist_pytorch.init_dist_training_env(config)

    check.check_config(config)

    dist_pytorch.barrier()
    cpm_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # get the tokenizer
    base_path = os.path.abspath(os.path.dirname(__file__))
    tokenizer = GPT2Tokenizer(
        os.path.join(base_path, 'dataloaders', config.tokenizer_path,
                     config.tokenizer_vocab_file),
        os.path.join(base_path, 'dataloaders', config.tokenizer_path,
                     config.tokenizer_vocab_model))
    train_dataloader, _ = load_data(config, 'train', tokenizer, 1)
    eval_dataloader, _ = load_data(config, 'valid', tokenizer, 1)
    print(f"train_dataset length:{len(train_dataloader.dataset)}")
    print(f"train length:{len(train_dataloader)}")
    print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
    print(f"eval length:{len(eval_dataloader)}")

    evaluator = Evaluator(config, eval_dataloader)
    training_state = TrainingState()
    # trainer = Trainer(config, training_event, evaluator, training_state, device=device)
    trainer = Trainer(driver=cpm_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)

    training_state._trainer = trainer

    dist_pytorch.barrier()
    trainer.init()

    dist_pytorch.barrier()
    init_evaluation_start = time.time()
    training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(
        trainer)
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_loss=training_state.eval_avg_loss,
        eval_embedding_average=training_state.eval_embedding_average,
        time=init_evaluation_end - init_evaluation_start)
    # training_event.on_init_evaluate(init_evaluation_info)
    cpm_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state, init_evaluation_info["time"]

    # training_event.on_init_end()
    cpm_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier()
    epoch = -1
    # training_event.on_train_begin()
    cpm_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
    # training_event.on_train_end()
    cpm_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config, state = main()

    if not dist_pytorch.is_main_process():
        exit()

    e2e_time = time.time() - now
    training_perf = (dist_pytorch.global_batch_size(config) *
                     state.global_steps) / state.raw_train_time
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
