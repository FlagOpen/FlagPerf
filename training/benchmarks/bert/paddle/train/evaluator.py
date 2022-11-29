from concurrent.futures import ProcessPoolExecutor

# import torch
# import torch.distributed as dist

from dataloaders.dataloader import WorkerInitializer, create_eval_dataloader
from train.driver import distributed
import config

import paddle
import paddle.distributed as dist


class Evaluator:

    def __init__(
        self,
        eval_dir: str,
        global_batch_size: int,
        max_steps: int,
        worker_init: WorkerInitializer = None,
    ):
        self.eval_dir = eval_dir

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

            self._dataloader = create_eval_dataloader(
                config.eval_dir, config.eval_batch_size,
                config.max_predictions_per_seq, config.num_eval_examples,
                self.worker_init)

        return self._dataloader

    @paddle.no_grad()
    def evaluate(self, trainer):
        self.eval_count += 1

        eval_dataloader = self.fetch_dataloader()

        trainer.model.eval()

        total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
        total_masked = 0

        for batch in eval_dataloader:

            loss, mlm_acc, num_masked = trainer.inference(batch)
            total_eval_loss += loss * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked
            distributed.barrier()
        trainer.model.train()

        if dist.is_initialized():
            dist.all_reduce(total_eval_mlm_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_eval_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_masked, op=dist.ReduceOp.SUM)
        print('---------------------------------------------')
        print(total_eval_loss)
        print(total_eval_mlm_acc)
        print('---------------------------------------------')
        # Average by number of examples
        total_eval_mlm_acc /= total_masked
        total_eval_loss /= total_masked

        return total_eval_loss.item(), total_eval_mlm_acc.item()
