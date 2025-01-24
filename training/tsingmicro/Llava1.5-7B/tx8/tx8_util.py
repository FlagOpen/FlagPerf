import torch
import torch.distributed as dist
from transformers import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
from torch.utils.data import DistributedSampler

def _all_reduce_by_group(input_, groups = None):
    ruduce_data = xm.all_reduce(xm.REDUCE_SUM, input_, groups=groups, pin_layout=True)
    return ruduce_data


class IdentityToAllReduce(torch.autograd.Function):
    all_reduce_sharding_groups = None

    @staticmethod
    def load(all_reduce_sharding_groups):
        IdentityToAllReduce.all_reduce_sharding_groups = all_reduce_sharding_groups

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce_by_group(grad_output, groups = IdentityToAllReduce.all_reduce_sharding_groups)

class AllReduceToIdentity(torch.autograd.Function):
    all_reduce_sharding_groups = None

    @staticmethod
    def load(all_reduce_sharding_groups):
        AllReduceToIdentity.all_reduce_sharding_groups = all_reduce_sharding_groups

    @staticmethod
    def forward(ctx, input_):
        return _all_reduce_by_group(input_, groups = IdentityToAllReduce.all_reduce_sharding_groups)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def all_reduce_sharding_group_load(all_reduce_sharding_groups = None):
    IdentityToAllReduce.load(all_reduce_sharding_groups)
    AllReduceToIdentity.load(all_reduce_sharding_groups)

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, num_tp_ranks=2, num_dp_ranks=2,shuffle=False,drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.num_tp_ranks = num_tp_ranks
        self.num_dp_ranks = num_dp_ranks
        self.drop_last = drop_last

    def __iter__(self):
        data_len = len(self.dataset)
        indices = None

        if not self.drop_last: 
            floor_data_len =   (data_len // self.num_tp_ranks) * self.num_tp_ranks
            # indices = list(range(self.rank // self.num_tp_ranks, floor_data_len, self.num_tp_ranks))
            indices = list(range(self.rank % self.num_tp_ranks, floor_data_len, self.num_tp_ranks))
        else:
            tp_local_rank = self.rank % self.num_tp_ranks
            dp_global_rank = self.rank // self.num_tp_ranks
            num_samples_per_tp = data_len // (self.num_tp_ranks * self.num_dp_ranks)
            remainder = data_len % (self.num_tp_ranks * self.num_dp_ranks)
            
            start = num_samples_per_tp * tp_local_rank + min(tp_local_rank, remainder)
            end = start + num_samples_per_tp + (1 if tp_local_rank < remainder else 0)
            
            total_samples = end - start
            num_samples_per_dp = total_samples // self.num_dp_ranks
            remainder = total_samples % self.num_dp_ranks
            
            start_dp = num_samples_per_dp * dp_global_rank + min(dp_global_rank, remainder)
            end_dp = start_dp + num_samples_per_dp + (1 if dp_global_rank < remainder else 0)
        
            indices = list(range(start_dp, end_dp))

        return iter(indices)

    def __len__(self):
        total_samples = len(self.dataset) // (self.num_tp_ranks * self.num_dp_ranks)
        return total_samples // self.num_dp_ranks