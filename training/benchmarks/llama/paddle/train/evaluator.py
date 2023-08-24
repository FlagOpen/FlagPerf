import os
import sys
import pdb
import paddle
import time
import math
import numpy as np
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_paddle

class Evaluator:
    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader

    def evaluate(self, trainer, config):
        model = trainer.model
        world_size = dist_paddle.get_world_size()

        model.eval()
        all_losses = []
        if config.eval_iters > 0:
            consumed_samples = (
                ((trainer.global_steps) // config.eval_steps - 2)
                * config.eval_iters
                * config.per_device_eval_batch_size
                * config.world_size
            )
            consumed_samples = max(consumed_samples, 0)
            with paddle.no_grad():
                for iters, inputs in enumerate(self.eval_dataloader):
                    if consumed_samples <= iters < consumed_samples + config.eval_iters:
                        outputs = model(**inputs)
                        loss = outputs[0]
                        all_losses.append(loss)
                        iters += 1
            all_losses = paddle.to_tensor(all_losses)
            eval_loss = all_losses.mean()
            if paddle.distributed.is_initialized():
                # Collect total scores from all ranks
                paddle.distributed.all_reduce(eval_loss,
                                            op=paddle.distributed.ReduceOp.SUM)

            # Average by number of examples
            eval_loss /= world_size
            return eval_loss.item()

    

        