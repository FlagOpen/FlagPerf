import os

import torch
from torch.types import Device

from driver import dist_pytorch

class Evaluator:
    """Evaluator"""
    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader
        self.device = config.device

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def evaluate(self, trainer):
        model = trainer.model
        model.eval()

        total_output = 0.0
        num_examples = 0.0
        with torch.no_grad():
            # For all the batches in the dataset.
            for step, inputs in enumerate(self.eval_dataloader):
                # Forward pass through the model.
                inputs = self.process_batch(inputs, self.device)
                output = model(**inputs)

                # For accuracy, return the number of correctly predicted samples.
                mask = inputs['labels'] != -100
                outputs = torch.argmax(output['logits'], -1)[mask]
                num_examples += outputs.shape.numel()
                correct = (outputs == inputs['labels'][mask]).float()
                output = correct.sum()

                total_output += output
        acc = total_output / num_examples
        return acc.item()
