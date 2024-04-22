import torch

def adjust_token_id(token_id):
    
    replacement_map = {
        338.0: 32.0,
        380.0: 33.0,
        339.0: 34.0,
        414.0: 35.0
    }
    
    return replacement_map.get(token_id, token_id)

def evaluator(pred, y):
    gt = float(y[0][0][1])
    predict = pred[:,-1,:]
    answer = float(torch.argmax(predict, dim=1))
    answer = adjust_token_id(answer)
    if answer == gt:
        return 1
    else:
        return 0
