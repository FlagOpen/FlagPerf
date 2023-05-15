from timm.scheduler import create_scheduler_v2, scheduler_kwargs


def create_scheduler(optimizer, updates_per_epoch, args):
    """Build the learning rate scheduler."""
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    return lr_scheduler, num_epochs
