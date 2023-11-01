"""Swin Transformer Pretraining"""

import os
import sys
import time
from typing import Any, Tuple

import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders import build_loader
from models import create_model
from train.trainer_adapter import create_optimizer
from schedulers import create_scheduler
from utils.utils import NativeScalerWithGradNormCount

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    logger = model_driver.logger
    init_start_time = logger.previous_log_time

    init_helper.set_seed(config.seed, config.vendor)
    
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    
    evaluator = Evaluator(data_loader_val)

    training_state = TrainingState()

    model = create_model(config)
    model.to(config.device)
    optimizer = create_optimizer(model, config)
    model = trainer_adapter.model_to_ddp(model)
    loss_scaler = NativeScalerWithGradNormCount()
    lr_scheduler = create_scheduler(config, optimizer, len(data_loader_train))
    
    trainer = Trainer(driver=model_driver,
                    training_state=training_state,
                    device=config.device,
                    config=config,
                    )
    training_state._trainer = trainer
    
    if not config.do_train:
        return config, training_state
    
    if config.aug_mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.model_label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.model_label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)

    epoch = -1
    max_accuracy = 0.0

    train_start_time = time.time()
    
    for epoch in range(config.train_start_epoch, config.train_epochs):
        training_state.epoch = epoch
        data_loader_train.sampler.set_epoch(epoch)
        trainer.train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        acc1, acc5, loss = evaluator.evaluate(config, model)
        max_accuracy = max(max_accuracy, acc1)
        
        training_state.eval_acc1, training_state.eval_acc5, training_state.eval_loss = acc1, acc5, loss
        training_state.max_accuracy = max_accuracy
        eval_result = dict(
                    eval_loss=training_state.eval_loss,
                    eval_acc1=training_state.eval_acc1,
                    eval_acc5=training_state.eval_acc5,
                    max_accuracy=training_state.max_accuracy)
        trainer.driver.event(Event.EVALUATE, eval_result)
    
    end_training_state = trainer.detect_training_status(training_state)
    model_driver.event(Event.TRAIN_END)

    training_state.raw_train_time =  time.time() - train_start_time

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    global_batch_size = dist_pytorch.global_batch_size(config_update)
    e2e_time = time.time() - start
    finished_info = {"e2e_time": e2e_time}
    if config_update.do_train:
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
            "train_no_eval_time": state.no_eval_time,
            "pure_training_computing_time": state.pure_compute_time,
            "throughput(ips)_raw": state.num_trained_samples / state.raw_train_time,
            "throughput(ips)_no_eval":
            state.num_trained_samples / state.no_eval_time,
            "throughput(ips)_pure_compute":
            state.num_trained_samples / state.pure_compute_time,
        }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
