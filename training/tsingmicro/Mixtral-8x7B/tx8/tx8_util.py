import os
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
            indices = list(range(self.rank // self.num_tp_ranks, floor_data_len, self.num_tp_ranks)) #rank 0=1
            # indices = list(range(self.rank % self.num_tp_ranks, floor_data_len, self.num_tp_ranks)) #rank 0=2
            # print(f"---indices={indices[0:5]},self.rank{self.rank}----")
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

def process_bar(current, total, prefix='', auto_rm=True):
    bar = '=' * int(current / total * 50)
    bar = f' {prefix} |{bar.ljust(50)}| ({current}/{total}) {current / total:.1%} | '
    print(bar, end='\r', flush=True)
    if auto_rm and current == total:
        print(end=('\r' + ' ' * len(bar) + '\r'), flush=True)

global layer_index
layer_index = 0
def _init_with_torchdistX(module, split_weight_path = None, num_hidden_layers = 1):
    from torchdistx import deferred_init
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    def check_fn(k):
        return not isinstance(k, FSDP)
    deferred_init.materialize_module(module, check_fn=check_fn)
    global layer_index
    for name, param in module.named_parameters():
        if 'flat_param_' not in name:
            if layer_index >= num_hidden_layers:
                weight_name = f"{split_weight_path}/{name}.pt"
            else:
                weight_name = f"{split_weight_path}/{name}_{layer_index}.pt"
            assert os.path.exists(weight_name) , f"!!!!!! weight_name:{weight_name} is not exists!"
            dtype = param.dtype
            load_weight = torch.load(weight_name)
            param.data = load_weight
            param = param.to(dtype=dtype)
    layer_index += 1
    process_bar(layer_index, num_hidden_layers+1, 'Weight Loading', auto_rm=False)