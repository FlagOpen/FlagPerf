import torch

def cross_entropy(outputs, target):
    #para:   outputs, [b, s, vocab_size]
    #        target, [b, s]
    #return: loss, [b, s] 

    logits = outputs.clone()
    # logits = outputs
    logits_max = torch.max(logits, dim=-1)[0]

    # Subtract the maximum value.
    logits.sub_(logits_max.unsqueeze(dim=-1))
    # Sum of exponential of logits along vocab dimension across all GPUs.
    exp_logits = logits.exp()
    sum_exp_logits = exp_logits.sum(dim=-1)

    logits_2d = logits.view(-1, logits.size()[-1])
    target_1d = target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                            device=logits_2d.device)
    predit_ligits_1d = logits_2d[arange_1d, target_1d]
    predit_ligits = predit_ligits_1d.view_as(target)

    loss = torch.log(sum_exp_logits) - predit_ligits

    return loss
