# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库
# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录
# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time

    seed = config.seed

    init_helper.set_seed(seed, model_driver.config.vendor)

    train_dataset = build_train_dataset(config)
    eval_dataset = build_eval_dataset(config)
    train_dataloader = build_train_dataloader(train_dataset, config)
    eval_dataloader = build_eval_dataloader(eval_dataset, config)

    # 根据 eval_dataloader 构建evaluator
    evaluator = Evaluator(config, eval_dataloader)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    # evaluation统计
    init_evaluation_start = time.time()  # evaluation起始时间，单位为秒
    all_c, all_top1, all_top5 = evaluator.evaluate(trainer.model,
                                                   trainer.device)

    init_evaluation_end = time.time()  # evaluation结束时间，单位为秒
    # time单位为秒
    init_evaluation_info = dict(time=init_evaluation_end -
                                init_evaluation_start)
    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    # init 统计
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    training_state.init_time = (init_end_time -
                                init_start_time) / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = time.time()  # 训练起始时间，单位为ms

    # 训练过程
    epoch = -1
    while training_state.global_steps < config.max_steps and \
            not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)

    # 训练时长，单位为秒
    training_state.raw_train_time = time.time() - raw_train_start_time

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - start

    training_perf = (dist_pytorch.global_batch_size(config_update) *
                     state.global_steps) / state.raw_train_time
    finished_info = {
        "e2e_time": e2e_time,
        "training_images_per_second": training_perf,
        "converged": state.converged,
        "final_accuracy": state.eval_mAP,
        "raw_train_time": state.raw_train_time,
        "init_time": state.init_time,
        "num_trained_samples": state.num_trained_samples,
        "pure_training_computing_time": state.pure_compute_time,
        "throughput(ips)_raw": state.num_trained_samples / state.raw_train_time,
        "throughput(ips)_no_eval": state.num_trained_samples / state.no_eval_time,
        "throughput(ips)_pure_compute": state.num_trained_samples / state.pure_compute_time,
    }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
