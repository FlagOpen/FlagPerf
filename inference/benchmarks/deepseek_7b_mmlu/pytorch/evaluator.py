
def evaluator(pred, y, dataloader):
    tokenizer = dataloader.tokenizer
    gt = y[0][0][1]
    predict = pred[:,-1,:]
    answer = torch.argmax(predict, dim=1)
    answer_str = tokenizer.decode(answer)
    valid_answers=['A','B','C','D']
    answer_str=''.join([c for c in answer_str if c in valid_answers])
    gt_str = tokenizer.decode(gt)
    if answer_str == gt_str:
        return 1
    else:
        return 0