import torch


def evaluator(pred, y, dataloader):

    tokenizer = dataloader.dataset.tokenizer

    gt = y[0][0][1]
    gt_str = tokenizer.decode(gt)
    
    answer_str = tokenizer.decode(pred)
    valid_answers = ['A', 'B', 'C', 'D']
    answer_str = ''.join([c for c in answer_str if c in valid_answers])
    
    if answer_str == gt_str:
        return 1
    else:
        return 0