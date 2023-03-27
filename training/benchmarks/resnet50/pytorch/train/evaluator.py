import torch
import torch.distributed as dist

class Evaluator:
    """Evaluator"""

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.args = args
        self.total_loss = 0.0
        self.total_acc1 = 0.0
        self.total_acc5 = 0.0
        self.total_batch = 0

    def __update(self, loss, acc1, acc5):
        self.total_loss += loss
        self.total_acc1 += acc1
        self.total_acc5 += acc5
        self.total_batch += 1

    def evaluate(self, trainer):
        """evaluate"""
        self.total_loss, self.total_acc1, self.total_acc5 = 0.0, 0.0, 0.0
        self.total_batch = 0
        with torch.no_grad():
            for _, batch in enumerate(self.dataloader):
                batch = trainer.process_batch(batch, self.args.device)
                loss, acc1, acc5 = trainer.inference(batch)
                self.__update(loss.item(), acc1.item(), acc5.item())

        if dist.is_available() and dist.is_initialized():
            total = torch.tensor([self.total_loss, self.total_acc1, self.total_acc5, self.total_batch], 
                                            dtype=torch.float32, device=self.args.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            self.total_loss, self.total_acc1, self.total_acc5, self.total_batch = total.tolist()

        loss = self.total_loss / self.total_batch
        acc1 = self.total_acc1 / self.total_batch
        acc5 = self.total_acc5 / self.total_batch
        return loss, acc1, acc5
