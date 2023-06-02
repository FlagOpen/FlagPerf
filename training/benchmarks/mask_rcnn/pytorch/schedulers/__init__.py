from torch.optim.lr_scheduler import MultiStepLR


def create_scheduler(optimizer, args):
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=args.lr_steps,
                               gamma=args.lr_gamma)
    return lr_scheduler
