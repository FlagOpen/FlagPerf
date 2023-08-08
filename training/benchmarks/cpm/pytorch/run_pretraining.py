"""CPM Pretraining"""
import os
import sys
import time

import config
from dataloaders.tokenization_gpt2 import GPT2Tokenizer
from dataloaders.dataloader import load_data
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from train import trainer_adapter

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Event, dist_pytorch
from driver.helper import InitHelper

logger = None


def main():
    global logger
    global config

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    init_helper = InitHelper(config)
    cpm_driver = init_helper.init_driver(globals(), locals())
    logger = cpm_driver.logger
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    cpm_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time
    init_helper.set_seed(config.seed, config.vendor)
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
    trainer = Trainer(driver=cpm_driver,
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
    training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(
        trainer)
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_loss=training_state.eval_avg_loss,
        eval_embedding_average=training_state.eval_embedding_average,
        time=init_evaluation_end - init_evaluation_start)
    cpm_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state

    cpm_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier(config.vendor)

    cpm_driver.event(Event.TRAIN_START)
    train_start_time = time.time()
    epoch = 0
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
        epoch += 1
    cpm_driver.event(Event.TRAIN_END)
    training_state.raw_train_time = time.time() - train_start_time
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config_updated, state = main()

    if not dist_pytorch.is_main_process():
        exit()

    e2e_time = time.time() - now
    training_perf = (dist_pytorch.global_batch_size(config_updated) *
                     state.global_steps) / state.raw_train_time
    if config_updated.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_avg_loss,
            "final_mlm_accuracy": state.eval_embedding_average,
            "init_time": state.init_time,
            "raw_train_time": state.raw_train_time,
            "train_no_eval_time": state.no_eval_time,
            "pure_training_computing_time": state.pure_compute_time,
            "throughput(ips)_raw":
            state.num_trained_samples / state.raw_train_time,
            "throughput(ips)_no_eval":
            state.num_trained_samples / state.no_eval_time,
            "throughput(ips)_pure_compute":
            state.num_trained_samples / state.pure_compute_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
