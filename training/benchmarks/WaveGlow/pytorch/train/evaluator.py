import time
import torch

from contextlib import contextmanager
from utils.utils import reduce_tensor


class Evaluator(object):

    @torch.no_grad()
    def validate(self, model, criterion, epoch, batch_iter, world_size,
                 distributed_run, batch_to_gpu, amp_run, val_loader):
        """Handles all the validation scoring and printing"""
        with evaluating(model), torch.no_grad():
            val_loss = 0.0
            num_iters = 0
            val_items_per_sec = 0.0
            for i, batch in enumerate(val_loader):
                torch.cuda.synchronize()
                iter_start_time = time.perf_counter()

                x, y, num_items = batch_to_gpu(batch)
                #AMP upstream autocast
                with torch.cuda.amp.autocast(enabled=amp_run):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                if distributed_run:
                    reduced_val_loss = reduce_tensor(loss.data,
                                                     world_size).item()
                    reduced_num_items = reduce_tensor(num_items.data, 1).item()
                else:  #
                    reduced_val_loss = loss.item()
                    reduced_num_items = num_items.item()
                val_loss += reduced_val_loss

                torch.cuda.synchronize()
                iter_stop_time = time.perf_counter()
                iter_time = iter_stop_time - iter_start_time

                items_per_sec = reduced_num_items / iter_time
                self.logger.log(step=(epoch, batch_iter, i),
                                data={'val_items_per_sec': items_per_sec})
                val_items_per_sec += items_per_sec
                num_iters += 1

            val_loss = val_loss / num_iters
            val_items_per_sec = val_items_per_sec / num_iters

            self.logger.log(step=(epoch, ), data={'val_loss': val_loss})
            self.logger.log(step=(epoch, ),
                            data={'val_items_per_sec': val_items_per_sec})

            return val_loss, val_items_per_sec


@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()
