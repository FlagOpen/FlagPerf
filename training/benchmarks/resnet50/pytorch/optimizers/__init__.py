from torch.optim import SGD


def create_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = SGD(params,
              lr=args.lr,
              momentum=args.momentum,
              weight_decay=args.weight_decay)
    return opt
