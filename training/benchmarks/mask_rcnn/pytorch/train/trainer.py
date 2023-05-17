import math
import time
import torch
from torch.types import Device
import os
import sys

from model import create_model


from schedulers import create_scheduler
from optimizers import create_optimizer

import utils.train.train_eval_utils as utils
from train.evaluator import Evaluator
from train.training_state import TrainingState

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch
from utils.train import save_on_master


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None
        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None

    def init(self):
        config = self.config

        pretrain_path = os.path.join(config.data_dir, config.pretrained_path)
        coco_weights_pretrained_path = os.path.join(
            config.data_dir, config.coco_weights_pretrained_path)

        dist_pytorch.main_proc_print(
            f"pretrain_path:{pretrain_path}, coco_weights_pretrained_path:{coco_weights_pretrained_path}"
        )
        self.model = create_model(self.config)
        self.model.to(self.device)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)
        # Attention: remember to move model to device before create_optimizer, otherwise, you will get a RuntimeError:
        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! when resuming training
        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler()

    def train_one_epoch(self,
                        dataloader,
                        eval_dataloader,
                        epoch,
                        train_loss: list,
                        learning_rate: list,
                        val_map: list,
                        det_results_file: str,
                        seg_results_file: str,
                        print_freq=50,
                        warmup=True,
                        scaler=None):

        state = self.training_state
        driver = self.driver
        device = self.device
        model = self.model
        optimizer = self.optimizer
        config = self.config
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        state.epoch += 1
        dist_pytorch.main_proc_print(f"state.epoch: {state.epoch}")
        mean_loss, lr = utils.train_one_epoch(model,
                                              optimizer,
                                              dataloader,
                                              device,
                                              epoch,
                                              state=state,
                                              config=self.config,
                                              print_freq=print_freq,
                                              warmup=warmup,
                                              scaler=scaler)

        # update learning rate
        self.lr_scheduler.step()

        # evaluate after every epoch
        det_info, seg_info = utils.evaluate(model, eval_dataloader, device)

        if det_info is not None:
            state.eval_mAP = det_info[0]
            print(f"training_state.eval_mAP:{state.eval_mAP}")

        if seg_info is not None:
            state.eval_segMAP = seg_info[0]
            print(f"training_state.eval_segMAP:{state.eval_segMAP}")

        # 只在主进程上进行写操作
        if config.local_rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            val_map.append(state.eval_mAP)  # pascal mAP

            # 写det结果
            with open(det_results_file, "a") as f:
                # 写入的数据包括coco指标，还有loss和learning rate
                result_info = [
                    f"{i:.4f}" for i in det_info + [mean_loss.item()]
                ] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # 写seg结果
            with open(seg_results_file, "a") as f:
                # 写入的数据包括coco指标, 还有loss和learning rate
                result_info = [
                    f"{i:.4f}" for i in seg_info + [mean_loss.item()]
                ] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        if config.output_dir:
            # 只在主进程上执行保存权重操作
            model_without_ddp = model
            if config.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[config.gpu])
                model_without_ddp = model.module

            train_state = {
                'model': model_without_ddp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'train_start_ts': state.train_start_timestamp,
            }
            if config.amp:
                train_state["scaler"] = self.grad_scaler.state_dict()

            checkpoint_path = os.path.join(config.output_dir, "checkpoint",
                                           f'model_{epoch}.pth')
            save_on_master(train_state, checkpoint_path)

        driver.event(Event.EPOCH_END, state.epoch)
        # check training state
        self.detect_training_status()
        return mean_loss, lr

    def detect_training_status(self):
        state = self.training_state
        config = self.config
        if state.eval_mAP >= config.target_mAP and state.eval_segMAP >= config.target_segMAP:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_mAP: {state.eval_mAP}, target_mAP: {config.target_mAP}"
            )
            state.converged_success()

        if state.epoch > config.max_epochs:
            state.end_training = True

        return state.end_training