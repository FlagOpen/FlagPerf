import torch
import torchvision.models as models


def create_model(args) -> torch.nn.Module:
    """create model"""
    if args.pretrained:
        print(f"=> using pre-trained model {args.name}")
        model = models.__dict__[args.name](pretrained=True)
    else:
        print(f"=> creating model {args.name}")
        model = models.__dict__[args.name]()
    return model
