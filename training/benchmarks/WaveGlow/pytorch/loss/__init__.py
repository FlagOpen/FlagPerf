from . import loss_function


def create_criterion(args):
    try:
        sigma = args.sigma
    except AttributeError:
        sigma = None
    criterion = loss_function.get_loss_function(args.name, sigma)
    return criterion
