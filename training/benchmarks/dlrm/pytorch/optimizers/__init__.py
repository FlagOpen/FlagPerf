import torch


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