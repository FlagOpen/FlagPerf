def topk(output, target, ks=(1, )):
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def evaluator(pred, ground_truth):
    top1, top5 = topk(pred, ground_truth, ks=(1, 5))
    return top1
