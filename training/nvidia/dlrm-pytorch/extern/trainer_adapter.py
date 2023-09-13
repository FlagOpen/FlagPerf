import itertools
import torch
import numpy as np

from apex import parallel


class CudaGraphWrapper:

    def __init__(self,
                 model,
                 train_step,
                 parallelize,
                 zero_grad,
                 cuda_graphs=False,
                 warmup_steps=20):

        self.cuda_graphs = cuda_graphs
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

        if cuda_graphs:
            self.stream = torch.cuda.Stream()
        else:
            # if not using graphs, parallelize the model immediately
            # otherwise do this in the warmup phase under the graph stream
            self.model = self._parallelize(self.model)
            self.stream = torch.cuda.default_stream()

    def _copy_input_data(self, *train_step_args):
        if len(train_step_args) != len(self.static_args):
            raise ValueError(
                f'Expected {len(self.static_args)} arguments to train step'
                f'Got: {len(train_step_args)}')

        for data, placeholder in zip(train_step_args, self.static_args):
            if placeholder is None:
                continue
            placeholder.copy_(data)

    def _cuda_graph_capture(self, *train_step_args):
        self._copy_input_data(*train_step_args)
        self.graph = torch.cuda.CUDAGraph()
        self._zero_grad(self.model)
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss = self._train_step(self.model, *self.static_args)
        return self.loss

    def _cuda_graph_replay(self, *train_step_args):
        self._copy_input_data(*train_step_args)
        self.graph.replay()

    def _warmup_step(self, *train_step_args):
        with torch.cuda.stream(self.stream):
            if self.step == 0:
                self.model = self._parallelize(self.model)
                self.static_args = list(train_step_args)
            else:
                self._copy_input_data(*train_step_args)

            self._zero_grad(self.model)
            self.loss = self._train_step(self.model, *self.static_args)
            return self.loss

    def train_step(self, *train_step_args):
        self.step += 1

        if not self.cuda_graphs:
            self._zero_grad(self.model)
            self.loss = self._train_step(self.model, *train_step_args)
            return self.loss

        if self.step == 0:
            self.stream.wait_stream(torch.cuda.current_stream())

        if self.step < self.warmup_iters:
            return self._warmup_step(*train_step_args)

        if self.graph is None:
            torch.cuda.synchronize()
            self._cuda_graph_capture(*train_step_args)

        self._cuda_graph_replay(*train_step_args)
        return self.loss


def get_cuda_graph_wrapper(model, config, embedding_optimizer, mlp_optimizer,
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
        rank = config.local_rank
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

    trainWrapper = CudaGraphWrapper(
        model,
        forward_backward,
        parallelize,
        zero_grad,
        cuda_graphs=config.cuda_graphs,
    )

    return trainWrapper


