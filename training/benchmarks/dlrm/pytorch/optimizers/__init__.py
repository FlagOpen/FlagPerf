import torch
from apex import optimizers as apex_optim

from driver import dist_pytorch


def create_mlp_optimizer(model, args):
    data_parallel_lr = args.lr
    world_size = args.n_device
    if args.Adam_MLP_optimizer:
        MLP_model_parallel_lr = args.lr
    else:
        MLP_model_parallel_lr = args.lr / world_size

    if dist_pytorch.is_main_process():
        mlp_params = [{
            'params': list(model.top_model.parameters()),
            'lr': data_parallel_lr
        }, {
            'params': list(model.bottom_model.mlp.parameters()),
            'lr': MLP_model_parallel_lr
        }]
    else:
        mlp_params = [{
            'params': list(model.top_model.parameters()),
            'lr': data_parallel_lr
        }]

    if args.Adam_MLP_optimizer:
        mlp_optimizer = apex_optim.FusedAdam(mlp_params)
    else:
        mlp_optimizer = apex_optim.FusedSGD(mlp_params)
    return mlp_optimizer


def create_embedding_optimizer(model, args):
    # DDP introduces a gradient average through allreduce(mean), which doesn't apply to bottom model.
    # Compensate it with further scaling lr
    world_size = args.n_device
    if args.Adam_embedding_optimizer:
        embedding_model_parallel_lr = args.lr
    else:
        embedding_model_parallel_lr = args.lr / world_size

    embedding_params = [{
        'params':
        list(model.bottom_model.embeddings.parameters()),
        'lr':
        embedding_model_parallel_lr
    }]

    if args.Adam_embedding_optimizer:
        embedding_optimizer = torch.optim.SparseAdam(embedding_params)
    else:
        embedding_optimizer = torch.optim.SGD(embedding_params)
    return embedding_optimizer