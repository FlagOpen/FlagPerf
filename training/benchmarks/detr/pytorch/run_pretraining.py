"""detr Pretraining"""
# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库
import torch
from torch.utils.data import DataLoader, DistributedSampler

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))  
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

# 本地库
import config
import util.misc as utils
from models import build_model
from driver import Event, dist_pytorch
from driver.helper import InitHelper

# 导入相关的模块、方法、变量
from dataloaders import build_dataset, get_coco_api_from_dataset
from train import trainer_adapter
from train.training_state import TrainingState
from train.evaluator import Evaluator
from train.trainer import Trainer

logger = None


def main()-> Tuple[Any, Any]:
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

    #logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time

    # 根据厂商设置全局随机种子
    init_helper.set_seed(config.seed, config.vendor)
    
    # 设置模型
    model, criterion, postprocessors = build_model(config)
    model.to(config.device)

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device])
        model_without_ddp = model.module

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)


    # 设置数据集
    dataset_train = build_dataset(image_set='train', args=config)
    dataset_val = build_dataset(image_set='val', args=config)

    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.train_batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.eval_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=config.num_workers)
    base_ds = get_coco_api_from_dataset(dataset_val)


    # 创建TrainingState对象
    training_state = TrainingState()
    
    # 验证器
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    evaluator = Evaluator(base_ds, iou_types)
    
    # 训练器
    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,    
                      training_state=training_state,
                      device=config.device,
                      config=config)
    training_state._trainer = trainer

    if not config.do_train:
        return config, training_state
    
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    # 开始训练
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time

    # 训练过程
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            sampler_train.set_epoch(epoch)
        trainer.train_one_epoch(model, criterion, data_loader_train, 
            optimizer, config.device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        trainer.evaluate(model, criterion, postprocessors,
            data_loader_val, base_ds, config.device, epoch)
    
    # 训练结束
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
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
    global_batch_size = dist_pytorch.global_batch_size(config_update)
    training_perf = (global_batch_size *
                         state.global_steps) / state.raw_train_time
    if config_update.do_train: 
        finished_info = {
            "e2e_time": e2e_time,
            "init_time": state.init_time,
            "raw_train_time": state.raw_train_time,
            "training_images_per_second": training_perf,
            "converged": state.converged,
            "final_mAP": state.eval_mAP,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
