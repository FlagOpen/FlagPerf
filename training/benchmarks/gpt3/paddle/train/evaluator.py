import os
import sys
import pdb
import paddle
import time
import math
import numpy as np
from train.driver import dist_paddle
import paddle.nn as nn
from tqdm import tqdm
from icecream import ic

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))

class Evaluator:

    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader

    def evaluate(self, trainer):
        model = trainer.model
        world_size = dist_paddle.get_world_size()

        model.eval()
        all_losses = []
        with paddle.no_grad():
            for _, inputs in enumerate(self.eval_dataloader):
                outputs = model(**inputs)
                loss = outputs[0].mean().detach()
                all_losses.append(loss.item())
   
        all_losses_tensor = paddle.to_tensor(np.mean(all_losses),
                                             dtype=paddle.float32)

        if paddle.distributed.is_initialized():
            # Collect total scores from all ranks
            paddle.distributed.all_reduce(all_losses_tensor,
                                          op=paddle.distributed.ReduceOp.SUM)

        # Average by number of examples
        all_losses_tensor /= world_size
        return all_losses_tensor.item()

    

        