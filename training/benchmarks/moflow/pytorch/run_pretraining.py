# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库
import torch

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录
# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

# 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from runtime.logger import MetricsLogger, PerformanceLogger, setup_logging
from runtime.common import get_newest_checkpoint, load_state
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.dataloader import build_datasets, build_train_dataloader

logger = None

torch._C._jit_set_autocast_mode(True)


def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    args = model_driver.config
    dist_pytorch.init_dist_training_env(args)
    dist_pytorch.barrier(args.vendor)
    model_driver.event(Event.INIT_START)

    torch.cuda.set_stream(torch.cuda.Stream())
    os.makedirs(args.results_dir, exist_ok=True)

    args.distributed = dist_pytorch.get_world_size() > 1
    # logger
    logger = model_driver.logger
    dllogger = setup_logging(args)
    world_size = args.n_device
    perf_logger, acc_logger = None, None
    if args.local_rank == 0:
        perf_logger = PerformanceLogger(dllogger,
                                        args.train_batch_size * world_size,
                                        args.warmup_steps)
        acc_logger = MetricsLogger(dllogger)

    train_dataset, _ = build_datasets(args)
    train_dataloader = build_train_dataloader(args, train_dataset)

    if args.save_epochs == -1:
        args.save_epochs = args.epochs
    if args.eval_epochs == -1:
        args.eval_epochs = args.epochs
    if args.steps == -1:
        args.steps = args.epochs * len(train_dataloader)

    seed = args.seed
    init_helper.set_seed(seed, args.vendor)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    evaluator = Evaluator()
    trainer = Trainer(
        driver=model_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=args.device,
        args=args,
        perf_logger=perf_logger,
        acc_logger=acc_logger,
        train_dataloader=train_dataloader,
    )
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(args.vendor)
    trainer.init()
    dist_pytorch.barrier(args.vendor)

    if not args.do_train:
        return args, training_state

    model_driver.event(Event.INIT_END)

    # TRAIN_START
    dist_pytorch.barrier(args.vendor)
    model_driver.event(Event.TRAIN_START)
    train_start_time = time.time()

    snapshot_path = get_newest_checkpoint(args.results_dir)
    dist_pytorch.main_proc_print(f"snapshot_path: {snapshot_path}")

    first_epoch, step = 0, 0
    if snapshot_path is not None:
        snapshot_epoch, ln_var = load_state(snapshot_path,
                                            trainer.model_callable,
                                            optimizer=trainer.optimizer,
                                            device=args.device)
        trainer.loss_callable.ln_var = torch.nn.Parameter(torch.tensor(ln_var))
        first_epoch = snapshot_epoch + 1
        step = first_epoch * len(train_dataloader)
    else:
        first_epoch = 0
        step = 0


    if first_epoch > args.epochs:
        dist_pytorch.main_proc_print(
            f'Model was already trained for {first_epoch} epochs, skip pretraining'
        )

    # 训练过程
    epoch = first_epoch
    while epoch < args.epochs:
        training_state.epoch = epoch
        step = trainer.train_one_epoch(train_dataloader, step)
        epoch += 1
        if step >= args.steps:
            break
    # TRAIN_END事件
    training_state.train_time = time.time() - train_start_time
    model_driver.event(Event.TRAIN_END)

    if args.local_rank == 0:
        # The same report for each epoch
        acc_logger.summarize(step=tuple())
        perf_logger.summarize(step=tuple())

        res = evaluator.evaluate(args, trainer.config, acc_logger)
        dist_pytorch.main_proc_print(f"evaluate results: {res}")
        training_state.nuv = res['nuv']
        if training_state.nuv >= args.target_nuv:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_nuv: {training_state.nuv}, target_nuv: {args.target_nuv}"
            )
            training_state.converged_success()

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:

        finished_info = {
            "e2e_time":
            e2e_time,
            "train_time":
            state.train_time,
            "train_no_eval_time":
            state.no_eval_time,
            "pure_training_computing_time":
            state.pure_compute_time,
            "throughput(ips)_raw":
            round(state.num_trained_samples / state.train_time, 2),
            "throughput(ips)_no_eval":
            round(state.num_trained_samples / state.no_eval_time, 2),
            "throughput(ips)_pure_compute":
            round(state.num_trained_samples / state.pure_compute_time, 2),
            "converged":
            state.converged,
            "final_nuv":
            state.nuv,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
