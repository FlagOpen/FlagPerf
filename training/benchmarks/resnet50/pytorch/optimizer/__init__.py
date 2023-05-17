from torch.optim import SGD
from torch.optim import Optimizer

def create_optimizer(model, args) -> Optimizer:
    optimizer = SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay_rate,
    )
    return optimizer
