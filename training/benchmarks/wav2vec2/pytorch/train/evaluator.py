import torch
import torch.distributed as dist
from common.helpers import to_gpu
from common import tb_dllogger as logger


class Evaluator:

    def __init__(self, dataloader):
        self.dataloader = dataloader

    @torch.no_grad()
    def validate(self, epoch, step, model, criterion, val_metrics,
                 val_ema_metrics, world_size, fp16, bf16):

        val_losses = []
        val_acc = []
        val_wer = []
        ema_model = None
        valid_loader = self.dataloader
        for model, metrics, scope in [(model, val_metrics, 'val'),
                                      (ema_model, val_ema_metrics, 'val_ema')]:
            if model is None:
                continue

            model.eval()
            criterion.eval()
            metrics._start_accumulating(None, True, scope=scope)
            output_keys = None

            assert len(valid_loader) > 1, (
                'Validation needs at least 2 iterations to handle empty batches.'
            )

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
            logger.log(() if epoch is None else (epoch, ),
                       metrics,
                       scope=scope,
                       tb_iter=step)

            val_losses.append(metrics.metrics[scope]['loss'])
            val_acc = metrics.metrics[scope]['accuracy']
            if 'wer' in metrics.metrics[scope]:
                val_wer.append(metrics.metrics[scope]['wer'])
            model.train()
            criterion.train()
        return val_losses, val_acc, val_wer
