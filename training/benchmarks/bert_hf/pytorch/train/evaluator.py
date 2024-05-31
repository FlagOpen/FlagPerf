import torch


class Evaluator:
    """Evaluator"""

    def __init__(self):
        pass

    def accuracy(self, output, input_ids, labels):
        pred = torch.argmax(output.logits, dim=2)

        mask = input_ids == 103
        masked_pred = pred[mask]
        masked_label = labels[mask]
        correct = masked_pred[masked_pred == masked_label]

        return len(correct) / len(masked_label)
