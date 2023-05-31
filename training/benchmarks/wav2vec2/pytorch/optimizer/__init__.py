
from common.fairseq.optim.fused_adam import get_fused_adam_class
from common.utils import print_once

def create_optimizer(model, args):

    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'adam' and not (args.fp16 or args.bf16):
        print_once('WARNING: Using FusedAdam instead of Adam')
        kw.update({'betas': args.adam_betas, 'eps': args.adam_eps})
        fused_adam_cls = get_fused_adam_class()
        print(fused_adam_cls, "fused_adam_cls")
        optimizer = fused_adam_cls(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    return optimizer
