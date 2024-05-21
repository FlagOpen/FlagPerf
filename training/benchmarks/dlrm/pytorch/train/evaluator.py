import torch

from driver import dist_pytorch as dist
import utils.utils as utils
from dataloaders.utils import prefetcher


class Evaluator:
    """Test distributed DLRM model
    Args:
        model (DistDLRM):
        eval_dataloader (torch.utils.data.DataLoader):
    """

    def __init__(self, config, eval_dataloader):
        self.config = config
        self.eval_dataloader = eval_dataloader

    def evaluate(self, model, eval_dataloader):
        """
        data_loader: validation dataloader
        """
        config = self.config
        device = config.base_device
        world_size = config.n_device

        model.eval()

        batch_sizes_per_gpu = [
            config.eval_batch_size // world_size for _ in range(world_size)
        ]
        test_batch_size = sum(batch_sizes_per_gpu)

        if config.eval_batch_size != test_batch_size:
            print(f"Rounded test_batch_size to {test_batch_size}")

        # Test bach size could be big, make sure it prints
        default_print_freq = max(524288 * 100 // test_batch_size, 1)
        print_freq = default_print_freq if config.print_freq is None else config.print_freq

        steps_per_epoch = len(eval_dataloader)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            'step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))

        with torch.no_grad():
            timer = utils.StepTimer()

            # ROC can be computed per batch and then compute AUC globally, but I don't have the code.
            # So pack all the outputs and labels together to compute AUC. y_true and y_score naming follows sklearn
            y_true = []
            y_score = []
            data_stream = torch.cuda.Stream()

            batch_iter = prefetcher(iter(eval_dataloader), data_stream)
            loss_fn = torch.nn.BCELoss(reduction="mean")

            timer.click(synchronize=(device == 'cuda'))
            for step in range(len(eval_dataloader)):
                numerical_features, categorical_features, click = next(
                    batch_iter)
                torch.cuda.synchronize()

                last_batch_size = None
                if click.shape[0] != test_batch_size:  # last batch
                    last_batch_size = click.shape[0]
                    padding_size = test_batch_size - last_batch_size

                    if numerical_features is not None:
                        padding_numerical = torch.empty(
                            padding_size,
                            numerical_features.shape[1],
                            device=numerical_features.device,
                            dtype=numerical_features.dtype)
                        numerical_features = torch.cat(
                            (numerical_features, padding_numerical), dim=0)

                    if categorical_features is not None:
                        padding_categorical = torch.ones(
                            padding_size,
                            categorical_features.shape[1],
                            device=categorical_features.device,
                            dtype=categorical_features.dtype)
                        categorical_features = torch.cat(
                            (categorical_features, padding_categorical), dim=0)

                with torch.cuda.amp.autocast(enabled=config.amp):
                    output = model(numerical_features, categorical_features,
                                   batch_sizes_per_gpu)
                    output = output.squeeze()
                    output = output.float()

                if world_size > 1:
                    output_receive_buffer = torch.empty(test_batch_size,
                                                        device=device)
                    torch.distributed.all_gather(
                        list(output_receive_buffer.split(batch_sizes_per_gpu)),
                        output)
                    output = output_receive_buffer

                if last_batch_size is not None:
                    output = output[:last_batch_size]

                if config.auc_device == "CPU":
                    click = click.cpu()
                    output = output.cpu()

                y_true.append(click)
                y_score.append(output)

                timer.click(synchronize=(device == 'cuda'))

                if timer.measured is not None:
                    metric_logger.update(step_time=timer.measured)
                    if step % print_freq == 0 and step > 0:
                        metric_logger.print(
                            header=f"Test: [{step}/{steps_per_epoch}]")

            if dist.is_main_process():
                y_true = torch.cat(y_true)
                y_score = torch.sigmoid(torch.cat(y_score)).float()
                auc = utils.roc_auc_score(y_true, y_score)
                loss = loss_fn(y_score, y_true).item()
                print(f'test loss: {loss:.8f}', )
            else:
                auc = None
                loss = None

            if world_size > 1:
                torch.distributed.barrier()

        model.train()

        return auc, loss
