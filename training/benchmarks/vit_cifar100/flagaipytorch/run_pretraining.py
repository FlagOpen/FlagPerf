"""Vit Cifar100 Pretraining"""

import time
import os
import sys
import argparse

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100

from flagai.env_trainer import EnvTrainer
from flagai.env_args import EnvArgs
from flagai.auto_model.auto_loader import AutoLoader
from training_state import TrainingState

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
import driver
from driver import Driver, Event, dist_flagaipytorch, check

logger = None
global vit_trainer


def load_cifar_dataset(train_dataset_path, eval_dataset_path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR100(root=train_dataset_path, train=True, download=False, transform=transform_train)
    test_dataset = CIFAR100(root=eval_dataset_path, train=False, download=False, transform=transform_test)
    return train_dataset, test_dataset


lr = 2e-5
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    # if trainer.fp16:
    #    images = images.half()
    labels = [b[1] for b in batch]
    labels = torch.tensor(labels).long()
    return {"images": images, "labels": labels}


def validate(logits, labels, meta=None):
    _, predicted = logits.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    accuracy = correct / total
    return accuracy


def main():
    import config
    from config import mutable_params
    global logger

    if 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    config.device = dist_flagaipytorch.get_device()
    config.n_device = dist_flagaipytorch.get_ndevice()
    config.dist_backend = 'nccl'
    
    vit_driver = Driver(config, config.mutable_params)
    vit_driver.setup_config(argparse.ArgumentParser("VIT"))
    vit_driver.setup_modules(driver, globals(), locals())

    logger = vit_driver.logger
    dist_flagaipytorch.init_dist_training_env(config)
    check.check_config(config, "")
    vit_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    # prepare data and model
    # model_dir = os.path.join(config.data_dir, "model")
    loader = AutoLoader(task_name="classification",
                        model_name="vit-base-p16-224",
                        #model_dir="model_dir",
                        num_classes=100)

    model = loader.get_model()
    train_dataset, val_dataset = load_cifar_dataset(config.train_data, config.eval_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.max_epochs)

    vit_env_args = EnvArgs(
        env_type="pytorch",
        experiment_name="vit-cifar100",
        batch_size=config.batch_size,
        num_gpus=config.n_device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.max_epochs,
        log_interval=config.log_freq,
        eval_interval=config.eval_interval,
        load_dir=None,
        pytorch_device=dist_flagaipytorch.get_device(),
        save_dir=None,
        save_interval=1000,
        num_checkpoints=1,
    )

    vit_env_args.add_arg(arg_name="data_dir", default="", type=str)
    vit_env_args.add_arg(arg_name="extern_config_dir", default="", type=str )
    vit_env_args.add_arg(arg_name="extern_config_file", default="", type=str )
    vit_env_args.add_arg(arg_name="extern_module_dir", default="", type=str )
    vit_env_args.add_arg(arg_name="enable_extern_config", default=False, type=None, store_true=True)
    vit_env_args = vit_env_args.parse_args()
    vit_trainer = EnvTrainer(vit_env_args)

    vit_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    vit_training_state = TrainingState()
    vit_training_state.init_time = (init_end_time - init_start_time) / 1e+3

    vit_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    vit_trainer.train(model,
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  train_dataset=train_dataset,
                  valid_dataset=val_dataset,
                  metric_methods=[["accuracy", validate]],
                  collate_fn=collate_fn)

    vit_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    vit_training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, vit_training_state


if __name__ == '__main__':
    now = time.time()
    config, state = main()

    e2e_time = time.time() - now
    finished_info = {
            "e2e_time": e2e_time,
            "converged": state.converged,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
    }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
