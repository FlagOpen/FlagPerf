import torch
import torch.distributed as dist
from torch.optim import Optimizer
import config

from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel as DDP


from common.fairseq.optim.adam import FairseqAdam
from common.fairseq.optim.fp16_optimizer import FP16Optimizer
from common.fairseq.optim.fused_adam import get_fused_adam_class
from common.utils import print_once


def convert_model(model: nn.Module) -> nn.Module:
    return model


def create_optimizer(model, args):

    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'adam' and (args.fp16 or args.bf16):

        print_once('WARNING: Using Fairseq FP16Optimizer')

        # based on fairseq.optim.FP16Optimizer.build_optimizer
        flatten = True  # not args.fp16_no_flatten_grads
        args.betas = args.adam_betas
        args.eps = args.adam_eps

        params = list(filter(lambda p: p.requires_grad, model.parameters()))

        fp32_params = FP16Optimizer.build_fp32_params(args, params,
                                                      flatten=flatten)

        # based on fairseq.optim.build_optimizer
        def build_optimizer(cfg, params, *extra_args, **extra_kwargs):
            if all(isinstance(p, dict) for p in params):
                params = [t for p in params for t in p.values()]
            params = list(filter(lambda p: p.requires_grad, params))
            return FairseqAdam(cfg, params, *extra_args, **extra_kwargs)

        if flatten:
            fp32_optimizer = build_optimizer(args, [fp32_params])
        else:
            fp32_optimizer = build_optimizer(args, fp32_params)

        if flatten and not fp32_optimizer.supports_flat_params:
            raise RuntimeError(
                f"chosen optimizer {fp32_optimizer.__class__.__name__} does "
                "not support flat params, please set --fp16-no-flatten-grads"
            )
        kwargs = {}
        optimizer = FP16Optimizer(args, params, fp32_optimizer, fp32_params,
                                  **kwargs)

    elif args.optimizer == 'adam' and not (args.fp16 or args.bf16):
        print_once('WARNING: Using FusedAdam instead of Adam')
        kw.update({'betas': args.adam_betas, 'eps': args.adam_eps})
        fused_adam_cls = get_fused_adam_class()
        print(fused_adam_cls,"fused_adam_cls")
        optimizer = fused_adam_cls(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    return optimizer

def model_to_fp16(model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[config.local_rank], find_unused_parameters=True)
        from common.fairseq.dist import ModuleProxyWrapper
        model = ModuleProxyWrapper(model)
    return model
