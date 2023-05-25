"""Mask R-CNN Pretraining"""
# 标准库
import datetime
import os
import sys
import time
from typing import Any, Tuple

# 三方库
import torch

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
import utils.train.train_eval_utils as utils

# 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from utils.train import mkdir
# 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader

logger = None


def main(start_ts) -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())

    config = model_driver.config

    # mkdir if necessary
    if config.output_dir:
        for sub_dir in ["checkpoint", "result", "plot"]:
            mkdir(os.path.join(config.output_dir, sub_dir))

    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)
    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    world_size = dist_pytorch.get_world_size(config.vendor)
    config.distributed = world_size > 1 or False

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = os.path.join(config.output_dir, "result",
                                    f"det_results_{world_size}_{now}.txt")
    seg_results_file = os.path.join(config.output_dir, "result",
                                    f"seg_results_{world_size}_{now}.txt")

    # 得到seed
    """
    这里获取seed的可行方式：
    1. 配置文件中的seed
    2. 自定义seed的生成方式：dist_pytorch.setup_seeds得到work_seeds数组，取其中某些元素。参考GLM-Pytorch的run_pretraining.py的seed生成方式
    3. 其他自定义方式
    """
    init_helper.set_seed(config.seed, config.vendor)

    # 构建dataset, dataloader 【train && validate】
    train_dataset = build_train_dataset(config)
    eval_dataset = build_eval_dataset(config)
    train_dataloader, train_sampler = build_train_dataloader(
        config, train_dataset)
    eval_dataloader = build_eval_dataloader(config, train_dataset,
                                            eval_dataset)

    # 根据 eval_dataloader 构建evaluator
    evaluator = Evaluator(config, eval_dataloader)

    # 创建TrainingState对象
    training_state = TrainingState()
    training_state.train_start_timestamp = start_ts

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
    """
    实现Evaluator 类的evaluate()方法，用于返回关键指标信息，如loss，eval_embedding_average等。
    例如：training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(trainer)
    """

    init_evaluation_end = time.time()  # evaluation结束时间，单位为秒
    """
    收集eval关键信息，用于日志输出
    例如： init_evaluation_info = dict(
        eval_loss=training_state.eval_avg_loss,
        eval_embedding_average=training_state.eval_embedding_average,
        time=init_evaluation_end - init_evaluation_start)
    """
    # time单位为秒
    init_evaluation_info = dict(time=init_evaluation_end -
                                init_evaluation_start)
    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
    model_without_ddp = trainer.model

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if config.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.

        # 读取之前保存的权重文件(包括优化器以及学习率策略)
        checkpoint = torch.load(config.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

        if "train_start_ts" in checkpoint:
            training_state.train_start_timestamp = checkpoint["train_start_ts"]
            dist_pytorch.main_proc_print(
                f"resume from checkpoint, read train_start_timestamp: {training_state.train_start_timestamp}"
            )

        if config.amp and "scaler" in checkpoint:
            trainer.grad_scaler.load_state_dict(checkpoint["scaler"])
        dist_pytorch.main_proc_print(
            f"resume training from checkpoint. checkpoint: {config.resume}, start_epoch:{config.start_epoch}"
        )

    # do evaluation
    if not config.do_train:
        return config, training_state

    # init计时
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    elapsed_ms = init_end_time - init_start_time
    training_state.init_time = elapsed_ms / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time  # 单位ms

    # 训练指标
    train_loss = []
    learning_rate = []
    val_map = []

    # 训练过程
    epoch = config.start_epoch
    while not training_state.end_training:
        if config.distributed:
            train_sampler.set_epoch(epoch)

        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader,
                                eval_dataloader,
                                epoch,
                                train_loss,
                                learning_rate,
                                val_map,
                                det_results_file,
                                seg_results_file,
                                print_freq=config.print_freq,
                                scaler=trainer.grad_scaler)

        epoch += 1

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time  # 训练结束时间，单位为ms

    # 训练时长，单位为秒
    raw_train_time_ms = int(raw_train_end_time - raw_train_start_time)
    training_state.raw_train_time = raw_train_time_ms / 1e+3

    return config, training_state


if __name__ == "__main__":

    start = time.time()
    updated_config, state = main(start)
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - state.train_start_timestamp
    if updated_config.do_train:
        # 构建训练所需的统计信息，包括不限于：e2e_time、training_samples_per_second、
        # converged、final_accuracy、raw_train_time、init_time
        training_perf = state.num_trained_samples / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "num_trained_samples": state.num_trained_samples,
            "training_samples_per_second": training_perf,
            "converged": state.converged,
            "final_map_bbox": state.eval_map_bbox,
            "final_map_segm": state.eval_map_segm,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
            "global_steps": state.global_steps,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
