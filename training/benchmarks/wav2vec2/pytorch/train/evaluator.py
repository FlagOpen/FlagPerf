import torch
import torch.distributed as dist
from common.helpers import to_gpu
from common import tb_dllogger as logger


class Evaluator:

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
        self.total_loss, self.total_acc1, self.total_acc5 = 0.0, 0.0, 0.0
        self.total_batch = 0
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                batch = trainer.process_batch(batch, self.args.device)
                loss, acc1, acc5 = trainer.inference(batch)
                self.__update(loss.item(), acc1.item(), acc5.item())

        if dist.is_available() and dist.is_initialized():
            total = torch.tensor([
                self.total_loss, self.total_acc1, self.total_acc5,
                self.total_batch
            ],
                                 dtype=torch.float32,
                                 device=self.args.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            self.total_loss, self.total_acc1, self.total_acc5, self.total_batch = total.tolist(
            )

        loss = self.total_loss / self.total_batch
        acc1 = self.total_acc1 / self.total_batch
        acc5 = self.total_acc5 / self.total_batch
        return loss, acc1, acc5


    @torch.no_grad()
    def validate(self, epoch, step, valid_loader, model, criterion,
                val_metrics, val_ema_metrics, world_size, fp16, bf16):

        val_losses = []
        val_acc = []
        val_wer=[]
        ema_model = None
        for model, metrics, scope in [(model, val_metrics, 'val'),
                                    (ema_model, val_ema_metrics, 'val_ema')]:
            if model is None:
                continue

            model.eval()
            criterion.eval()
            metrics._start_accumulating(None, True, scope=scope)
            output_keys = None

            assert len(valid_loader) > 1, (
                'Validation needs at least 2 iterations to handle empty batches.')

            for batch in valid_loader:
                is_empty_batch = len(batch) == 0
                if not is_empty_batch:
                    to_gpu(batch, fp16=fp16, bf16=bf16)

                    loss, _, logging_output = criterion(model, batch)

                    if output_keys is None:
                        output_keys = logging_output.keys()

                else:
                    assert output_keys is not None, (
                        f'Invalid iters num: {len(valid_loader)}')
                    logging_output = {k: 0 for k in output_keys}

                logging_output['ignore'] = int(is_empty_batch)
                metrics.log_scalars(logging_output)
                metrics.all_reduce(world_size)
                metrics.accumulate()

            metrics.finish_val(scope=scope)
            logger.log(() if epoch is None else (epoch,),  metrics, scope=scope,
                    tb_iter=step)
            
            val_losses.append(metrics.metrics[scope]['loss'])
            val_acc = metrics.metrics[scope]['accuracy']
            if 'wer' in metrics.metrics[scope]:
                val_wer.append(metrics.metrics[scope]['wer'])
            model.train()
            criterion.train()
        return val_losses, val_acc, val_wer