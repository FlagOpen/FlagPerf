from torch.optim import AdamW


def create_optimizer(model, args):
    opt = AdamW(model.parameters(), lr=args.lr)
    return opt
