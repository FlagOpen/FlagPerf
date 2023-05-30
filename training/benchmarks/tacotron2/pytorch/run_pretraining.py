"""Tacotron2 Pytorch Pretraining"""

# 标准库
import os
import sys
import time
from typing import Any, Tuple

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录
# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

# 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
# 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
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
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    init_helper.set_seed(config.seed, model_driver.config.vendor)

    world_size = dist_pytorch.get_world_size()
    config.distributed = world_size > 1 or config.multiprocessing_distributed

    # 构建dataset, dataloader 【train && validate】
    train_dataset = build_train_dataset(config)
    val_dataset = build_eval_dataset(config)
    train_dataloader = build_train_dataloader(
        config, train_dataset, distributed_run=config.distributed)
    val_dataloader = build_eval_dataloader(val_dataset, config)

    # 根据 eval_dataloader 构建evaluator
    evaluator = Evaluator(config, val_dataloader)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    trainer = Trainer(
        driver=model_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=config.device,
        config=config,
        world_size=world_size,
        train_dataloader=train_dataloader,
    )
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    # do evaluation
    if not config.do_train:
        return config, training_state

    # init 统计
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    training_state.init_time = (init_end_time -
                                init_start_time) / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time  # 训练起始时间，单位为ms

    # 训练过程
    while not training_state.end_training:
        trainer.train_one_epoch(train_dataloader)

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time  # 训练结束时间，单位为ms

    # 训练时长，单位为秒
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:
        training_perf = (dist_pytorch.global_batch_size(config_update) *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_samples_per_second": training_perf,
            "converged": state.converged,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
            "epoch": state.epoch,
            "global_steps": state.global_steps,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
