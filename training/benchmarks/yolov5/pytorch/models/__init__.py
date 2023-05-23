import os
from tkinter.ttk import Widget
import torch
import torchvision
from models.yolo import Model
from models.experimental import attempt_load
from utils.general import check_suffix,intersect_dicts
from utils.torch_utils import de_parallel,smart_DDP
from utils.downloads import attempt_download, is_url
import yaml
from pathlib import Path

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def create_model(config):
    # model = torchvision.models.mobilenet_v2()
    weights, cfg, resume = config.weights, config.cfg, config.resume

    # hpy = "path/to/hpy.yaml"
    # hyp = config.hyp
    # Hyperparameters
    if isinstance(config.hyp, str):
        with open(config.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print("-----hpy----:",hyp)
    device = config.device
    # nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # coco number of class: 80
    nc = 80
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    print("pretrained:",pretrained)
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        print("load model success!")
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    
    # # DDP mode
    # cuda = device.type != 'cpu'
    # if cuda and RANK != -1:
    #     model = smart_DDP(model)
    
    # Model attributes
    # nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # hyp['box'] *= 3 / nl  # scale to layers
    # hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # hyp['obj'] *= (config.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # hyp['label_smoothing'] = config.label_smoothing
    # model.nc = nc  # attach number of classes to model
    # model.hyp = hyp  # attach hyperparameters to model
    # # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    
    # # Read yaml (optional)
    # if isinstance(config.data, (str, Path)):
    #     with open(config.data, errors='ignore') as f:
    #         data = yaml.safe_load(f)  # dictionary
    # # coco class list
    # model.names = data["names"]
    
    return model

