import math
import time
import torch
import torch.utils.data
import torchvision
from torch.types import Device
import os
import sys

from model import create_model
from optimizers import create_optimizer
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState

from dataloaders.dataloader import get_coco, get_coco_kp, get_coco_api_from_dataset
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import utils.presets
import utils.utils

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    return utils.presets.DetectionPresetTrain(
    ) if train else utils.presets.DetectionPresetEval()


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

    def init(self):
        torch.set_num_threads(1)
        device = torch.device(self.config.device)
        dist_pytorch.main_proc_print("Init progress:")
        self.model = create_model()
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler()
        dataset, num_classes = get_dataset("coco", "train",
                                           get_transform(train=True),
                                           self.config.data_dir)
        dataset_test, _ = get_dataset("coco", "val",
                                      get_transform(train=False),
                                      self.config.data_dir)
        if self.config.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(dataset)
            self.test_sampler = torch.utils.data.SequentialSampler(
                dataset_test)

        if self.config.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(
                dataset, k=self.config.aspect_ratio_group_factor)
            self.train_batch_sampler = GroupedBatchSampler(
                self.train_sampler, group_ids, self.config.train_batch_size)
        else:
            self.train_batch_sampler = torch.utils.data.BatchSampler(
                self.train_sampler,
                self.config.train_batch_size,
                drop_last=True)

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.config.num_workers,
            collate_fn=utils.utils.collate_fn)

        self.data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.config.eval_batch_size,
            sampler=self.test_sampler,
            num_workers=self.config.num_workers,
            collate_fn=utils.utils.collate_fn)

        coco = get_coco_api_from_dataset(self.data_loader_test.dataset)
        self.evaluator = Evaluator(coco)

    def train_one_epoch(self):
        model = self.model
        optimizer = self.optimizer
        data_loader = self.data_loader
        device = self.device
        epoch = self.training_state.epoch
        if self.config.distributed:
            self.train_sampler.set_epoch(epoch)

        model.train()
        metric_logger = utils.utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            'lr', utils.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.utils.warmup_lr_scheduler(
                optimizer, warmup_iters, warmup_factor)

        for images, targets in metric_logger.log_every(data_loader, 100,
                                                       header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device)
                        for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            self.adapter.backward(losses, optimizer)

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        self.lr_scheduler.step()

        self.evaluate(self.model, self.data_loader_test, device=self.device)

        state = self.training_state
        config = self.config

        state.eval_mAP = self.evaluator.coco_eval['bbox'].stats.tolist()[0]
        if state.eval_mAP >= config.target_mAP:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_mAP: {state.eval_mAP}, target_mAP: {config.target_mAP}"
            )
            state.converged_success()

        if epoch >= config.max_epoch:
            state.end_training = True
        state.num_trained_samples += len(data_loader.dataset)

        return state.end_training

    @torch.no_grad()
    def evaluate(self, model, data_loader, device):
        coco = get_coco_api_from_dataset(data_loader.dataset)
        self.evaluator = Evaluator(coco)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        for images, targets in metric_logger.log_every(data_loader, 100,
                                                       header):
            images = list(img.to(device) for img in images)

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)

            outputs = [{k: v.to(cpu_device)
                        for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, outputs)
            }
            evaluator_time = time.time()
            self.evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time,
                                 evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        self.evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        self.evaluator.accumulate()
        self.evaluator.summarize()
        return self.evaluator
