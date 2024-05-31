from torch.optim import SGD


def create_optimizer(model, args):
    return SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
