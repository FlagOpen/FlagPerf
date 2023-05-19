from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler 
from utils.general import (one_cycle)

from dataloaders import hyp
def create_scheduler(optimizer, args):
    """Build the learning rate scheduler."""
    epochs = args.epochs
    
    if args.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    return scheduler
