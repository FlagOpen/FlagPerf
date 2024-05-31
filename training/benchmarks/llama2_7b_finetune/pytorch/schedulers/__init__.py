from torch.optim.lr_scheduler import StepLR


def create_scheduler(optimizer, train_config):
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    return scheduler
