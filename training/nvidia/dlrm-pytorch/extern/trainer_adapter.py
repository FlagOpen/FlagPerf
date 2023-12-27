from typing import Sequence
import itertools
import torch
import numpy as np

from apex import parallel
from apex import optimizers as apex_optim

from utils.distributed import is_distributed, is_main_process, get_rank
from utils.utils import LearningRateScheduler
from dataloaders.utils import get_embedding_sizes
from .model.network.embeddings import Embeddings, JointEmbedding, MultiTableEmbeddings, FusedJointEmbedding, JointSparseEmbedding
from .model.distributed import DistributedDlrm


class TrainerWrapper:

    def __init__(self,
                 model,
                 train_step,
                 parallelize,
                 zero_grad,
                 warmup_steps=20):

        self.warmup_iters = warmup_steps
        self.graph = None
        self.stream = None
        self.static_args = None

        self.model = model

        self._parallelize = parallelize
        self._train_step = train_step
        self._zero_grad = zero_grad

        self.loss = None
        self.step = -1
        self.model = self._parallelize(self.model)

    def train_step(self, *train_step_args):
        self.step += 1
        self._zero_grad(self.model)
        self.loss = self._train_step(self.model, *train_step_args)
        return self.loss


def get_trainer_wrapper(model, config, embedding_optimizer, mlp_optimizer,
                           loss_fn, grad_scaler):

    def parallelize(model):
        world_size = config.n_device
        use_gpu = "cpu" not in config.base_device.lower()
        if world_size <= 1:
            return model

        if use_gpu:
            model.top_model = parallel.DistributedDataParallel(model.top_model)
        else:  # Use other backend for CPU
            model.top_model = torch.nn.parallel.DistributedDataParallel(
                model.top_model)
        return model

    def forward_backward(model, *args):
        rank = get_rank()
        numerical_features, categorical_features, click = args
        world_size = config.n_device
        batch_sizes_per_gpu = [
            config.eval_batch_size // world_size for _ in range(world_size)
        ]
        batch_indices = tuple(
            np.cumsum([0] +
                      list(batch_sizes_per_gpu)))  # todo what does this do

        loss = None
        with torch.cuda.amp.autocast(enabled=config.amp):
            output = model(numerical_features, categorical_features,
                           batch_sizes_per_gpu).squeeze()
            loss = loss_fn(output,
                           click[batch_indices[rank]:batch_indices[rank + 1]])
        grad_scaler.scale(loss).backward()
        return loss

    def zero_grad(model):
        if config.Adam_embedding_optimizer or config.Adam_MLP_optimizer:
            model.zero_grad()
        else:
            # We don't need to accumulate gradient. Set grad to None is faster than optimizer.zero_grad()
            for param_group in itertools.chain(
                    embedding_optimizer.param_groups,
                    mlp_optimizer.param_groups):
                for param in param_group['params']:
                    param.grad = None

    trainWrapper = TrainerWrapper(
        model,
        forward_backward,
        parallelize,
        zero_grad,
    )

    return trainWrapper


def create_embeddings(embedding_type: str,
                      categorical_feature_sizes: Sequence[int],
                      embedding_dim: int,
                      device: str = "cuda",
                      hash_indices: bool = False,
                      fp16: bool = False) -> Embeddings:
    if embedding_type == "joint":
        return JointEmbedding(categorical_feature_sizes,
                              embedding_dim,
                              device=device,
                              hash_indices=hash_indices)
    elif embedding_type == "joint_fused":
        assert not is_distributed(), "Joint fused embedding is not supported in the distributed mode. " \
                                     "You may want to use 'joint_sparse' option instead."
        return FusedJointEmbedding(categorical_feature_sizes,
                                   embedding_dim,
                                   device=device,
                                   hash_indices=hash_indices,
                                   amp_train=fp16)
    elif embedding_type == "joint_sparse":
        return JointSparseEmbedding(categorical_feature_sizes,
                                    embedding_dim,
                                    device=device,
                                    hash_indices=hash_indices)
    elif embedding_type == "multi_table":
        return MultiTableEmbeddings(categorical_feature_sizes,
                                    embedding_dim,
                                    hash_indices=hash_indices,
                                    device=device)
    else:
        raise NotImplementedError(f"unknown embedding type: {embedding_type}")


def create_model(args, device, device_mapping, feature_spec):
    rank = get_rank()
    bottom_mlp_sizes = args.bottom_mlp_sizes if rank == device_mapping[
        'bottom_mlp'] else None
    world_embedding_sizes = get_embedding_sizes(
        feature_spec, max_table_size=args.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_sizes)
    # Embedding sizes for each GPU
    categorical_feature_sizes = world_categorical_feature_sizes[
        device_mapping['embedding'][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()

    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=num_numerical_features,
        hash_indices=args.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=args.top_mlp_sizes,
        interaction_op=args.interaction_op,
        fp16=args.amp,
        use_cpp_mlp=args.optimized_mlp,
        bottom_features_ordered=args.bottom_features_ordered,
        device=device)
    return model


def create_mlp_optimizer(model, args):
    data_parallel_lr = args.lr
    world_size = args.n_device
    if args.Adam_MLP_optimizer:
        MLP_model_parallel_lr = args.lr
    else:
        MLP_model_parallel_lr = args.lr / world_size

    if is_main_process():
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


def create_grad_scaler(args):
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp,
                                       growth_interval=int(1e9))
    return scaler



def create_scheduler(args, mlp_optimizer, embedding_optimizer):
    data_parallel_lr = args.lr
    world_size = args.n_device

    if args.Adam_MLP_optimizer:
        MLP_model_parallel_lr = args.lr
    else:
        MLP_model_parallel_lr = args.lr / world_size

    if is_main_process():
        mlp_lrs = [data_parallel_lr, MLP_model_parallel_lr]
    else:
        mlp_lrs = [data_parallel_lr]

    # DDP introduces a gradient average through allreduce(mean), which doesn't apply to bottom model.
    # Compensate it with further scaling lr
    if args.Adam_embedding_optimizer:
        embedding_model_parallel_lr = args.lr
    else:
        embedding_model_parallel_lr = args.lr / world_size

    embedding_lrs = [embedding_model_parallel_lr]

    base_lrs = [mlp_lrs, embedding_lrs]
    print("world_size", world_size, "create_scheduler base_lrs", base_lrs)

    lr_scheduler = LearningRateScheduler(
        optimizers=[mlp_optimizer, embedding_optimizer],
        base_lrs=[mlp_lrs, embedding_lrs],
        warmup_steps=args.warmup_steps,
        warmup_factor=args.warmup_factor,
        decay_start_step=args.decay_start_step,
        decay_steps=args.decay_steps,
        decay_power=args.decay_power,
        end_lr_factor=args.decay_end_lr / args.lr)

    return lr_scheduler