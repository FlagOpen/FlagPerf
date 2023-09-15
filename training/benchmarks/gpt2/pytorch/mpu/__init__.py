import torch

_DATA_PARALLEL_GLOBAL_RANKS = None
if torch.distributed.is_initialized():
    _DATA_PARALLEL_GLOBAL_RANKS = [i for i in range(torch.distributed.get_world_size())]

def get_data_parallel_rank():
    return torch.distributed.get_rank()

def get_data_parallel_world_size():
    return torch.distributed.get_world_size()