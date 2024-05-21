from transformers import get_scheduler


def create_scheduler(optimizer, train_dataloader, args):
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.max_epoch,
    )
    return lr_scheduler
