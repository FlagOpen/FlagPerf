from torch.optim import Adam


def create_optimizer(model, args):
    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    return optimizer