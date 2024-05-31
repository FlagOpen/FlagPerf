import os
import torch
import torch.distributed as dist
import config

from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from typing import Tuple



def model_to_fp16(model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def create_grad_scaler():
    return None

