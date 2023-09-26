import torch


def evaluator(pred, y):
    gt = float(y[0][0][1])
    predict = pred[:,-1,:]
    answer = float(torch.argmax(predict, dim=1))
    if answer == gt:
        return 1
    else:
        return 0
