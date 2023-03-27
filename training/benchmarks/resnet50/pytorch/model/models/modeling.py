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

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None and args.gpu != 0:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                # model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model.cuda()
    return model
