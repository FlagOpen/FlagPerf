# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from dataloaders.dataloader import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.torch_utils import select_device, time_sync

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.mp = 0.0
        self.mr = 0.0
        self.map50 = 0.0
        self.map = 0.0
        self.loss = torch.zeros(3)


    def process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    @torch.no_grad()
    def evaluate(self,
            data,
            weights=None,  # model.pt path(s)
            batch_size=32,  # batch size
            imgsz=640,  # inference size (pixels)
            conf_thres=0.001,  # confidence threshold
            iou_thres=0.6,  # NMS IoU threshold
            task='val',  # train, val, test, speed or study
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            workers=8,  # max dataloader workers (per RANK in DDP mode)
            single_cls=False,  # treat as single-class dataset
            augment=False,  # augmented inference
            verbose=False,  # verbose output
            save_txt=False,  # save results to *.txt
            save_hybrid=False,  # save label+prediction hybrid results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_json=False,  # save a COCO-JSON results file
            project=ROOT / 'runs/val',  # save to project/name
            name='exp',  # save to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            half=True,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            model=None,
            dataloader=None,
            save_dir=Path(''),
            plots=True,
            callbacks=Callbacks(),
            compute_loss=None,
            ):
        # Initialize/load model and set device
        training = model is not None
        if training:  # called by train.py
            device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

            half &= device.type != 'cpu'  # half precision only supported on CUDA
            model.half() if half else model.float()
        else:  # called directly
            device = select_device(device, batch_size=batch_size)

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            half &= (pt or jit or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
            if pt or jit:
                model.model.half() if half else model.model.float()
            elif engine:
                batch_size = model.batch_size
            else:
                half = False
                batch_size = 1  # export.py models default to batch-size 1
                device = torch.device('cpu')
                LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

            # Data
            data = check_dataset(data)  # check

        # Configure
        model.eval()
        is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
        nc = 1 if single_cls else int(data['nc'])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Dataloader
        if not training:
            model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz), half=half)  # warmup
            pad = 0.0 if task in ('speed', 'benchmark') else 0.5
            rect = False if task == 'benchmark' else pt  # square inference for benchmarks
            task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
            dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=rect,
                                        workers=workers, prefix=colorstr(f'{task}: '))[0]

        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1 = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0
        mp, mr, map50, map = self.mp, self.mr, self.map50, self.map
        loss = self.loss.to(device)
        jdict, stats, ap, ap_class = [], [], [], []
        pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            t1 = time_sync()
            if pt or jit or engine:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
            dt[1] += time_sync() - t2

            # Loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t3 = time_sync()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            dt[2] += time_sync() - t3

            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = self.process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        if not training:
            shape = (batch_size, 3, imgsz, imgsz)
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

        # Return results
        model.float()  # for training
        if not training:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
