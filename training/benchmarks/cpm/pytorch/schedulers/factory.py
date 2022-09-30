from .learning_rates import AnnealingLR

def create_scheduler(optimizer, args):
    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.max_steps

    num_iters = max(1, num_iters)
    init_step = -1
    if args.warmup_steps != 0:
        warmup_iter = args.warmup_steps
    else:
        warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.learning_rate,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler