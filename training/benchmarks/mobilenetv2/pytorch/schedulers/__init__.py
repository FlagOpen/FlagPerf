from torch.optim.lr_scheduler import StepLR


def create_scheduler(optimizer, args):
    lr_scheduler = StepLR(optimizer,
                          step_size=args.lr_steps,
                          gamma=args.lr_gamma)
    return lr_scheduler
