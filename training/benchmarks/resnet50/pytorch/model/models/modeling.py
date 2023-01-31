import torch
import torchvision.models as models


def create_model(args, ngpus_per_node)->torch.nn.Module:

    print(f"create_model args.pretrained: {args.pretrained}")
    print(f"create_model args.arch: {args.arch}")

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    print(f"create_model model: {model}")
    print(f"config distributed: {args.distributed}")
    print(f"config gpu: {args.gpu}")


    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print("create_model goes distributed")

        if torch.cuda.is_available():
            if args.gpu is not None and args.gpu != 0:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
                args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                print("create_model ddp")
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                # model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("create_model goes here")

            model.cuda()
            # model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=list(range(ngpus_per_node)))
            # model.to(model.device_ids[0])
    return model
