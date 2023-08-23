import os

import torch

from driver import dist_pytorch

class Evaluator:
    """Evaluator"""
    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader

    def evaluate(self, trainer):
        model = trainer.model
        model.eval()

        total_output = 0.0
        num_examples = len(self.eval_dataloader.dataset)
        with torch.no_grad():
            # For all the batches in the dataset.
            for step, inputs in enumerate(self.eval_dataloader):
                # Forward pass through the model.
                output = model(**inputs)
                # For accuracy, return the number of correctly predicted samples.
                outputs = torch.argmax(output, -1)
                correct = (outputs == inputs['labels']).float()
                output = correct.sum()

                # Reduce across processes.
                if dist_pytorch.is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(output)

                total_output += output
        acc = total_output / num_examples
        model.eval()
        return acc.item()