from gc import callbacks
import torch
from torch.types import Device
import torch.distributed as dist
import os
import sys
import time
import math
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import argparse
from pathlib import Path
import yaml

from models import create_model
from schedulers import create_scheduler
from utils.general import check_amp

from train.evaluator import Evaluator
from train.training_state import TrainingState

import config
from utils.general import LOGGER,labels_to_image_weights,check_img_size,one_cycle,check_dataset,init_seeds
from utils.torch_utils import EarlyStopping, ModelEMA, torch_distributed_zero_first
from utils.loss import ComputeLoss
from utils.callbacks import Callbacks
from utils.metrics import fitness
import train.val as val # for end-of-epoch mAP

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch
from dataloaders.dataloader import build_train_dataloader,build_eval_dataloader

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# 需要根据本项目做调整 todo
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# need to update
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
        self.global_batch_size = None
        self.overflow_buf = None
        

    def init(self):
        self.model = create_model(config)
        self.model = self._init_model(self.model, self.config, self.device)
        self.model = self.adapter.convert_model(self.model)
        # self.model = self.adapter.model_to_fp16(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)
        self.model = self.adapter.model_to_ddp(self.model)
        
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler()

    def _init_model(self, model, args, device):
        checkpoint_name = config.init_checkpoint
        if os.path.isfile(checkpoint_name):
            print('checkpoint_name', checkpoint_name)
            print('global rank {} is loading pretrained model {}'.format(
                dist_pytorch.get_rank(), checkpoint_name))
            # Load the checkpoint.
            checkpoint = torch.load(checkpoint_name, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

        model = model.to(device)
        return model

    def train_all_epoch(self,train_dataloader,train_dataset,val_dataloader):
        state = self.training_state
        
        config = self.config
        save_dir = Path(config.save_dir)
        
        # Hyperparameters
        # from utils.hyp_param import hpy
        if isinstance(config.hyp, str):
            with open(config.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        # amp = check_amp(self.model)  # check AMP
        amp = False
        
        # Config
        # init_seeds(config.seed + 1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = check_dataset(config.data)  # check /data/coco.yaml if None
        # train_path, val_path = data_dict['train'], data_dict['val']
       

        # Start training
        # t0 = time.time()
        nb = len(train_dataloader)  # number of batches

        nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        # nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
        nc = 80
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        start_epoch = 0
        self.lr_scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
        compute_loss = ComputeLoss(self.model)  # init loss class
        callbacks = Callbacks()
        callbacks.run('on_train_start')
         
        epochs = config.epochs
        
        # EMA
        ema = ModelEMA(self.model) if RANK in {-1, 0} else None
        plots = False
        
        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            callbacks.run('on_train_epoch_start')
            self.model.train()

            # Update image weights (optional, single-GPU only)
            if config.image_weights:
                cw = self.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=cw)  # image weights
                train_dataset.indices = random.choices(range(train_dataset.n), weights=iw, k=train_dataset.n)  # rand weighted idx

            device = config.device
            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(train_dataloader)
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            nb = len(train_dataloader)
            if RANK in {-1, 0}:
                pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            
            self.optimizer.zero_grad()
            
            # hard code
            nbs = 64
            batch_size = config.batch_size
            print("----batch size:",batch_size)
            gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
            imgsz = check_img_size(config.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
            
            if config.cos_lr:
                lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
            else:
                lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
            
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                callbacks.run('on_train_batch_start')
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if config.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = self.model(imgs)  # forward
                    # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    state.loss = loss
                    
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if config.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(self.optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(self.optimizer)  # optimizer.step
                    scaler.update()
                    self.optimizer.zero_grad()
                    if ema:
                        ema.update(self.model)
                    last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    state.lbox, state.lobj, state.lcls = mloss[0], mloss[1], mloss[2]
                    # mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    # pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                        # (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, plots)
                    if callbacks.stop_training:
                        return
                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
            self.lr_scheduler.step()

            # process 0
            if RANK in {-1, 0}:
                # mAP
                callbacks.run('on_train_epoch_end', epoch=epoch)
                ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                # final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
                final_epoch = (epoch + 1 == epochs) 
                if not config.noval or final_epoch:  # Calculate mAP
                    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
                    results, maps, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            half=amp,
                                            model=ema.ema,
                                            single_cls=config.single_cls,
                                            dataloader=val_dataloader,
                                            save_dir=save_dir,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)
                
                state.P,state.R, state.eval_mAP_0_5, state.eval_mAP_5_95 = results[0],results[1],results[2],results[3]

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                # stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
        print("---------end one epoch----------")
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------            
            
        
    def train_one_epoch_(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()
        epoch_start_num_sample = state.num_trained_samples

        for batch_idx, batch in enumerate(dataloader):

            state.global_steps += 1
            # TODO: Maybe we should update num_trained_samples after all epochs.
            state.num_trained_samples = state.global_steps * \
                dist_pytorch.global_batch_size(self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch)

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                images_per_second = (
                    dist_pytorch.global_batch_size(self.config) *
                    self.config.gradient_accumulation_steps) / step_total_time
                other_state["img/s"] = images_per_second
            if hasattr(self.optimizer, 'loss_scaler'):
                loss_scale = self.optimizer.loss_scaler.loss_scale
                other_state['loss_scale'] = loss_scale

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_loss, state.eval_acc1, state.eval_acc5 = self.evaluator.evaluate(
                    self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_loss=state.eval_loss,
                                   eval_acc1=state.eval_acc1,
                                   eval_acc5=state.eval_acc5,
                                   time=eval_end - eval_start)

            end_training = self.detect_training_status(state)
            step_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END,
                         message=step_info,
                         step=state.global_steps,
                         loss=state.loss)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if end_training:
                break

        epoch_start_num_sample += len(dataloader.dataset)
        state.num_trained_samples = epoch_start_num_sample

        self.lr_scheduler.step()
        driver.event(Event.EPOCH_END, state.epoch)
    
    def train_one_step(self, batch):
        # move data to the same device as model
        batch = self.process_batch(batch, self.config.device)
        state = self.training_state
        self.model.train()
        state.loss, state.acc1, state.acc5 = self.forward(batch)
        self.adapter.backward(state.global_steps, state.loss, self.optimizer)
        if dist.is_available() and dist.is_initialized():
            total = torch.tensor([state.loss, state.acc1, state.acc5],
                                 dtype=torch.float32,
                                 device=self.config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            total = total / dist.get_world_size()
            state.loss, state.acc1, state.acc5 = total.tolist()
        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer, self.grad_scaler)

    def detect_training_status(self, state):
        config = self.config
        # update yolov5's condition
        if state.eval_acc1 >= config.target_acc1:
            state.converged_success()

        if state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state):
        config = self.config
        do_eval = all([
            config.eval_data is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps %
            math.ceil(config.eval_interval_samples /
                      dist_pytorch.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination

    def forward(self, batch):
        images, target = batch
        output = self.model(images)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        # todo
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5

    def inference(self, batch):
        self.model.eval()
        output = self.forward(batch)
        return output

    def process_batch(self, batch, device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch

# train的时候需要用到此处的默认参数 
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()

