import torch


def evaluator(pred, x, y):
    mask = x == 103
    masked_pred = pred[mask]
    masked_y = y[mask]

    correct = masked_pred[masked_pred == masked_y]

    return len(correct), len(masked_y)
