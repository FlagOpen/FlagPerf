"""GPT Pretraining"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import time

import matplotlib.pyplot as plt
import paddle
import paddlenlp
from dataloaders.dataloader import load_data

# CURR_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_paddle
from paddlenlp.transformers import GPTTokenizer
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

logger = None


def main():
    import config
    from config import mutable_params

    global logger

    if config.use_env and "PADDLE_TRAINER_ID" in os.environ:
        config.local_rank = int(os.environ["PADDLE_TRAINER_ID"])

    gpt_driver = Driver(config, mutable_params)
    gpt_driver.setup_config(argparse.ArgumentParser("gpt"))
    gpt_driver.setup_modules(globals(), locals())

    logger = gpt_driver.logger

    dist_paddle.init_dist_training_env(config)
    dist_paddle.barrier()
    gpt_driver.event(Event.INIT_START)

    init_start_time = logger.previous_log_time
    paddlenlp.trainer.set_seed(config.seed, config)

    # get the tokenizer
    tokenizer = GPTTokenizer.from_pretrained(config.tokenizer_name_or_path)

    train_dataloader, eval_dataloader, test_dataloader = load_data(config, tokenizer)

    print(f"train_dataset length:{len(train_dataloader.dataset)}")
    print(f"train length:{len(train_dataloader)}")
    print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
    print(f"eval length:{len(eval_dataloader)}")

    evaluator = Evaluator(config, eval_dataloader)
    training_state = TrainingState()
    trainer = Trainer(
        driver=gpt_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=config.device,
        config=config,
    )

    training_state._trainer = trainer

    dist_paddle.barrier()
    trainer.init()

    dist_paddle.barrier()
    init_evaluation_start = time.time()
    training_state.eval_avg_loss = evaluator.evaluate(trainer)
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_loss=training_state.eval_avg_loss,
        time=init_evaluation_end - init_evaluation_start,
    )
    gpt_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state

    gpt_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e3

    dist_paddle.barrier()
    epoch = -1

    gpt_driver.event(Event.TRAIN_START)
    train_start_time = time.time()

    while (
        training_state.global_steps < config.max_steps
        and not training_state.end_training
    ):
        epoch += 1
        training_state.epoch = epoch
        loss = paddle.to_tensor(0.0)
        trainer.train_one_epoch(train_dataloader, loss)
    gpt_driver.event(Event.TRAIN_END)
    training_state.raw_train_time = time.time() - train_start_time

    return config, training_state, trainer.tr_loss


if __name__ == "__main__":
    now = time.time()
    config, state, tr_loss = main()

    if not dist_paddle.is_main_process():
        exit()

    e2e_time = time.time() - now
    training_perf = (
        dist_paddle.global_batch_size(config) * state.global_steps
    ) / state.raw_train_time
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_avg_loss,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

    # 可视化 loss
    # ic(trainer.tr_loss)

    plt.switch_backend("Agg")

    plt.figure()
    plt.plot(tr_loss, "b", label="loss")
    plt.ylabel("loss")
    plt.xlabel("perf_step")
    plt.legend()
    plt.savefig("./step_loss_dp.jpg")
