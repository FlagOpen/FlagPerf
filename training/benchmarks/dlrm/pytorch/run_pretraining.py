# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.dataloader import build_train_dataloader, build_eval_dataloader
from dataloaders.utils import get_embedding_sizes
from dataloaders.feature_spec import FeatureSpec
from utils.distributed import get_device_mapping, is_distributed

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
    config.distributed = dist_pytorch.get_world_size() > 1
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    feature_spec = load_feature_spec(config)

    cat_feature_count = len(get_embedding_sizes(feature_spec, None))
    validate_flags(config, cat_feature_count)

    world_embedding_sizes = get_embedding_sizes(
        feature_spec, max_table_size=config.max_table_size)
    device_mapping = get_device_mapping(world_embedding_sizes,
                                        num_gpus=dist_pytorch.get_world_size())

    train_dataloader = build_train_dataloader(config, feature_spec,
                                              device_mapping)
    eval_dataloader = build_eval_dataloader(config, feature_spec,
                                            device_mapping)

    seed = config.seed
    init_helper.set_seed(seed, model_driver.config.vendor)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    evaluator = Evaluator(config, eval_dataloader)
    trainer = Trainer(
        driver=model_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=config.device,
        config=config,
        device_mapping=device_mapping,
        feature_spec=feature_spec,
    )
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    if not config.do_train:
        return config, training_state

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    training_state.init_time = (init_end_time -
                                init_start_time) / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    train_start_time = time.time()
    training_state.train_time_start_ts = train_start_time

    # 训练过程
    epoch = 0
    while not training_state.end_training:
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
        epoch += 1

    # TRAIN_END事件
    training_state.train_time = time.time() - train_start_time
    model_driver.event(Event.TRAIN_END)

    return config, training_state


def load_feature_spec(args):
    if args.dataset_type == 'synthetic_gpu' and not args.synthetic_dataset_use_feature_spec:
        num_numerical = args.synthetic_dataset_numerical_features
        categorical_sizes = [
            int(s) for s in args.synthetic_dataset_table_sizes
        ]
        return FeatureSpec.get_default_feature_spec(
            number_of_numerical_features=num_numerical,
            categorical_feature_cardinalities=categorical_sizes)
    fspec_path = os.path.join(args.data_dir, args.feature_spec)
    return FeatureSpec.from_yaml(fspec_path)


def validate_flags(args, cat_feature_count):
    if args.max_table_size is not None and not args.hash_indices:
        raise ValueError(
            'Hash indices must be True when setting a max_table_size')

    if args.base_device == 'cpu':
        if args.embedding_type in ('joint_fused', 'joint_sparse'):
            print('WARNING: CUDA joint embeddings are not supported on CPU')
            args.embedding_type = 'joint'

        if args.amp:
            print('WARNING: Automatic mixed precision not supported on CPU')
            args.amp = False

        if args.optimized_mlp:
            print('WARNING: Optimized MLP is not supported on CPU')
            args.optimized_mlp = False

    if args.embedding_type == 'custom_cuda':
        if (not is_distributed()
            ) and args.embedding_dim == 128 and cat_feature_count == 26:
            args.embedding_type = 'joint_fused'
        else:
            args.embedding_type = 'joint_sparse'

    if args.embedding_type == 'joint_fused' and args.embedding_dim != 128:
        print(
            'WARNING: Joint fused can be used only with embedding_dim=128. Changed embedding type to joint_sparse.'
        )
        args.embedding_type = 'joint_sparse'

    if args.data_dir is None and (args.dataset_type != 'synthetic_gpu'
                                  or args.synthetic_dataset_use_feature_spec):
        raise ValueError(
            'Dataset argument has to specify a path to the dataset')

    # args.inference_benchmark_batch_sizes = [
    #     int(x) for x in args.inference_benchmark_batch_sizes
    # ]
    args.top_mlp_sizes = [int(x) for x in args.top_mlp_sizes]
    args.bottom_mlp_sizes = [int(x) for x in args.bottom_mlp_sizes]


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)
    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "num_trained_samples": state.num_trained_samples,
            "global_steps": state.global_steps,
            "train_time": state.train_time,
            "train_no_eval_time": state.no_eval_time,
            "pure_training_computing_time": state.pure_compute_time,
            "throughput(ips)_raw":
            state.num_trained_samples / state.train_time,
            "throughput(ips)_no_eval":
            state.num_trained_samples / state.no_eval_time,
            "throughput(ips)_pure_compute":
            state.num_trained_samples / state.pure_compute_time,
            "converged": state.converged,
            "final_auc": state.eval_auc,
            "best_auc": state.best_auc,
            "best_validation_loss": state.best_validation_loss,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
