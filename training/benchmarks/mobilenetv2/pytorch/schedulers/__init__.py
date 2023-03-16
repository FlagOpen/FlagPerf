from torch.optim.lr_scheduler import StepLR


def create_scheduler(optimizer, args):
    """Build the learning rate scheduler."""
    
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    return lr_scheduler
