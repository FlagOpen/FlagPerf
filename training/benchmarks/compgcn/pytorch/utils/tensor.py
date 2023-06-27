import torch
import torch.distributed as dist


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / num_gpus
    else:
        rt = torch.div(rt, num_gpus)
    return rt
