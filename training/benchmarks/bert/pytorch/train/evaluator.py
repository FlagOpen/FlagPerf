from concurrent.futures import ProcessPoolExecutor

import torch
import torch.distributed as dist

from dataloaders.dataloader import WorkerInitializer, create_eval_dataloader

import config
import os
import sys

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


class Evaluator:

    def __init__(self,
                 eval_dir: str,
                 proc_pool: ProcessPoolExecutor,
                 global_batch_size: int,
                 max_steps: int,
                 worker_init: WorkerInitializer = None,
                 use_cache: bool = False):
        self.eval_dir = eval_dir
        self.proc_pool = proc_pool
        self.use_cache = use_cache

        if worker_init is None:
            worker_init = WorkerInitializer.default()
        self.worker_init = worker_init

        self.eval_count = 0
        self.cached_batches = []

        self.need_next_training_shard = global_batch_size * max_steps > 10000
        self._dataloader = None
        self.fetch_dataloader()

    def fetch_dataloader(self):
        if self._dataloader is None:
            if self.need_next_training_shard:
                if self.eval_count == 0:
                    self.eval_dataset_future = self.proc_pool.submit(
                        create_eval_dataloader, config.eval_dir,
                        config.eval_batch_size, config.max_predictions_per_seq,
                        config.num_eval_examples, self.worker_init)
                else:
                    self._dataloader = self.eval_dataset_future.result(
                        timeout=None)
            else:
                self._dataloader = create_eval_dataloader(
                    config.eval_dir, config.eval_batch_size,
                    config.max_predictions_per_seq, config.num_eval_examples,
                    self.worker_init)

        return self._dataloader

    def evaluate(self, trainer):
        self.eval_count += 1

        eval_dataloader = self.fetch_dataloader()

        trainer.model.eval()

        total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
        total_masked = 0

        # on first eval, load and cache data on GPU
        if self.eval_count == 1 and self.use_cache:
            for batch in eval_dataloader:
                self.cached_batches.append(
                    [t.to(trainer.device) for t in batch])

        with torch.no_grad():
            for batch in self.cached_batches if self.use_cache else eval_dataloader:
                if not self.use_cache:
                    batch = [t.to(trainer.device) for t in batch]
                loss, mlm_acc, num_masked = trainer.inference(batch)
                total_eval_loss += loss * num_masked
                total_eval_mlm_acc += mlm_acc * num_masked
                total_masked += num_masked
                #torch.cuda.synchronize()
                dist_pytorch.barrier(config.vendor)
        trainer.model.train()

        if torch.distributed.is_initialized():
            # Collect total scores from all ranks
            torch.distributed.all_reduce(total_eval_mlm_acc,
                                         op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_eval_loss,
                                         op=torch.distributed.ReduceOp.SUM)
            total_masked = total_masked.to(torch.float32)
            torch.distributed.all_reduce(total_masked,
                                         op=torch.distributed.ReduceOp.SUM)

        # Average by number of examples
        total_eval_mlm_acc /= total_masked
        total_eval_loss /= total_masked

        return total_eval_loss.item(), total_eval_mlm_acc.item()
