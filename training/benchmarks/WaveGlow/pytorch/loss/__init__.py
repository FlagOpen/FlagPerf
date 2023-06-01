from . import loss_functions


def create_criterion(args):
    try:
        sigma = args.sigma
    except AttributeError:
        sigma = None
    criterion = loss_functions.get_loss_function(args.name, sigma)
    return criterion
