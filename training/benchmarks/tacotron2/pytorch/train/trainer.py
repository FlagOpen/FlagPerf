import torch
from torch.types import Device
import os
import sys
import time
import torch.distributed as dist
import numpy as np

from model import create_model, create_model_config
from schedulers import create_scheduler
from model.loss.loss_function import get_loss_function
from model.data.data_function import batch_to_gpu
from .utills import reduce_tensor

from train.evaluator import Evaluator
from train.training_state import TrainingState
import config

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 world_size, train_dataloader):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.model_config = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.criterion = None

        self.world_size = world_size
        self.train_dataloader = train_dataloader

    def init(self):
        self.model_config = create_model_config(config)
        self.model = create_model(config)
        self.model = self._init_model(self.model)
        self.model = self.adapter.model_to_ddp(self.model, self.config)
        self.criterion = get_loss_function()
        # self.model = self.adapter.model_to_fp16(self.model, self.optimizer)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)

        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.scaler = self.adapter.create_grad_scaler(self.config)

        torch.backends.cudnn.enabled = self.config.cudnn_enabled
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark

    def _init_model(self, model):
        # resume from checkpoint
        pass

        return model

    def train_one_epoch(self, train_dataloader):
        state = self.training_state
        driver = self.driver

        if self.config.local_rank == 0:
            state.epoch += 1

        driver.event(Event.EPOCH_BEGIN, state.epoch)

        torch.cuda.synchronize()
        # epoch_start_time = time.perf_counter()
        # used to calculate avg items/sec over epoch
        # reduced_num_items_epoch = 0

        # train_epoch_items_per_sec = 0.0

        # num_iters = 0
        # reduced_loss = 0

        if self.config.distributed:
            self.train_dataloader.sampler.set_epoch(state.epoch)

        epoch_start_num_sample = state.num_trained_samples

        for batch in train_dataloader:
            self.train_one_step(batch)

            self.detect_training_status()

            if state.end_training:
                break

        torch.cuda.synchronize()
        epoch_start_num_sample += len(train_dataloader.dataset)
        state.num_trained_samples = epoch_start_num_sample

        args = self.config

        # val_loss, val_items_per_sec = self.evaluator.evaluate(self)

        if (state.epoch % self.config.epochs_per_checkpoint
                == 0) and (args.bench_class == ""
                           or args.bench_class == "train"):
            save_checkpoint(self.model, self.optimizer, self.scaler,
                            self.training_state.epoch, self.model_config,
                            args.output, args.model_name, args.local_rank,
                            self.world_size)

        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, batch):
        driver = self.driver
        state = self.training_state
        args = self.config

        torch.cuda.synchronize()
        iter_start_time = time.perf_counter()
        adjust_learning_rate(self.training_state.global_steps,
                             self.training_state.epoch, self.optimizer,
                             args.learning_rate, args.lr_anneal_steps,
                             args.lr_anneal_factor, args.local_rank)

        self.model.zero_grad()
        x, y, num_items = batch_to_gpu(batch)

        # AMP upstream autocast
        with torch.cuda.amp.autocast(enabled=args.amp):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, self.world_size).item()
            reduced_num_items = reduce_tensor(num_items.data, 1).item()
        else:
            reduced_loss = loss.item()
            reduced_num_items = num_items.item()

        if np.isnan(reduced_loss):
            raise Exception("loss is NaN")

        # DLLogger.log(step=(epoch, i), data={'train_loss': reduced_loss})

        # accumulate number of items processed in this epoch
        # reduced_num_items_epoch += reduced_num_items

        if args.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           args.grad_clip_thresh)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           args.grad_clip_thresh)
            self.optimizer.step()

        self.model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        iter_stop_time = time.perf_counter()
        iter_time = iter_stop_time - iter_start_time
        items_per_sec = reduced_num_items / iter_time
        # train_epoch_items_per_sec += items_per_sec

        state.train_loss = reduced_loss
        step_info = dict(step=state.global_steps, train_loss=reduced_loss)

        self.training_state.global_steps += 1
        driver.event(Event.STEP_END, state.global_steps, message=step_info)

    def detect_training_status(self):
        config = self.config
        state = self.training_state
        if state.train_loss <= config.target_train_loss:
            state.converged_success()

        if state.epoch > config.max_epochs:
            state.end_training = True

        return state.end_training


# TODO 改成AnnealScheduler
def adjust_learning_rate(iteration, epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor, rank):
    p = 0

    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1**(p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor**p)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model, optimizer, scaler, epoch, config, output_dir,
                    model_name, local_rank, world_size):
    random_rng_state = torch.random.get_rng_state().cuda()
    cuda_rng_state = torch.cuda.get_rng_state(local_rank).cuda()

    random_rng_states_all = [
        torch.empty_like(random_rng_state) for _ in range(world_size)
    ]
    cuda_rng_states_all = [
        torch.empty_like(cuda_rng_state) for _ in range(world_size)
    ]

    if world_size > 1:
        dist.all_gather(random_rng_states_all, random_rng_state)
        dist.all_gather(cuda_rng_states_all, cuda_rng_state)
    else:
        random_rng_states_all = [random_rng_state]
        cuda_rng_states_all = [cuda_rng_state]

    random_rng_states_all = torch.stack(random_rng_states_all).cpu()
    cuda_rng_states_all = torch.stack(cuda_rng_states_all).cpu()

    if local_rank == 0:
        checkpoint = {
            'epoch': epoch,
            'cuda_rng_state_all': cuda_rng_states_all,
            'random_rng_states_all': random_rng_states_all,
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }

        checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        print("Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path))
        torch.save(checkpoint, checkpoint_path)

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(output_dir,
                                   "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(symlink_src, symlink_dst)


def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir,
                           "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""
