from torch.optim import lr_scheduler


def create_scheduler(optimizer, hyp, epochs):
    # Scheduler
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    return scheduler