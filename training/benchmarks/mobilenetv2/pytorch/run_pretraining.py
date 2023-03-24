"""Mobilenet V2 Pretraining"""

import os
import sys
import time
from typing import Any, Tuple


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  
from driver import Event, dist_pytorch
from driver.helper import InitHelper, get_finished_info

from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader

logger = None


def main() -> Tuple[Any, Any]:
    import config
    global logger
    

    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())  
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)


    logger = model_driver.logger
    init_start_time = logger.previous_log_time

    init_helper.set_seed(config.seed, config.vendor)

    train_dataset = build_train_dataset(config)
    eval_dataset = build_eval_dataset(config)
    train_dataloader = build_train_dataloader(train_dataset, config)
    eval_dataloader = build_eval_dataloader(eval_dataset, config)

    evaluator = Evaluator(config, eval_dataloader)

    training_state = TrainingState()

    trainer = Trainer(driver=model_driver,
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
    training_state.eval_loss, training_state.eval_acc1, training_state.eval_acc5 = evaluator.evaluate(trainer)

    init_evaluation_end = time.time() 
    init_evaluation_info = dict(
        eval_acc1=training_state.eval_acc1,
        eval_acc5=training_state.eval_acc5,
        time=init_evaluation_end - init_evaluation_start)
    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time 
    training_state.init_time = (init_end_time - init_start_time) / 1e+3 

    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time 

    epoch = -1
    while training_state.global_steps < config.max_steps and \
            not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)

    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time 

    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    global_batch_size = dist_pytorch.global_batch_size(config)
    e2e_time = time.time() - start
    finished_info = {"e2e_time": e2e_time}
    if config.do_train:
        training_perf = (global_batch_size *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_images_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_loss,
            "final_acc1": state.eval_acc1,
            "final_acc5": state.eval_acc5,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
