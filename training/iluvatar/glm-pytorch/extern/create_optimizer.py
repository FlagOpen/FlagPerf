from optimizers import FP16_Optimizer, get_optimizer_param_groups
from apex.optimizers import FusedAdam as Adam
from driver.dist_pytorch import main_proc_print


def create_optimizer(model, args):
    param_groups = get_optimizer_param_groups(model)
    optimizer = Adam(param_groups,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     betas=(args.adam_beta1, args.adam_beta2),
                     eps=args.adam_eps)
    main_proc_print(f'Optimizer = {optimizer.__class__.__name__}')
    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis
                                   })

    return optimizer
