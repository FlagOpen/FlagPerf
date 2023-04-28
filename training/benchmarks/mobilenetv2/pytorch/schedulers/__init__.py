from torch.optim.lr_scheduler import StepLR


def create_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    lr_scheduler = StepLR(optimizer,
                          step_size=args.lr_step_size,
                          gamma=args.lr_gamma)
    return lr_scheduler
