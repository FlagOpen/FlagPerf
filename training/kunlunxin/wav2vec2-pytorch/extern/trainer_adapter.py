import torch

def create_optimizer(model, args):

    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'adam' and not (args.fp16 or args.bf16):
        kw.update({'betas': args.adam_betas, 'eps': args.adam_eps})
        optimizer = torch.optim.Adam(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    return optimizer
