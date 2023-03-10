from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim import Optimizer

from model.fp16 import FP16_Module
from model.fp16 import FP16_Optimizer
from model.models import gpt2_get_params_for_weight_decay_optimization
from train.event.base import BaseTrainingEventInterface, CPM_MODEL
from apex.parallel import DistributedDataParallel as APEX_DDP
from converter import convert_model
from optimizers_adam import Adam


class ApexTrainingEvent(BaseTrainingEventInterface):

    def __init__(self, config):
        super(ApexTrainingEvent, self).__init__(config)
        self.model = None
        self.optimizer = None

        self.autocast_ctx = None

    def convert_model(self, model: CPM_MODEL) -> CPM_MODEL:
        return convert_model(model, self.config)

    def create_optimizer(self, model: CPM_MODEL) -> Optimizer:
        param_groups = gpt2_get_params_for_weight_decay_optimization(model)
        optimizer = Adam(param_groups,
                         lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay_rate)

        return optimizer

    def model_to_fp16(self, model: CPM_MODEL,
                      optimizer: Optimizer) -> Tuple[CPM_MODEL, Optimizer]:
        model = FP16_Module(model)
        args = self.config
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis
                                   })
        return model, optimizer

    def model_to_ddp(self, model: CPM_MODEL) -> CPM_MODEL:
        use_ddp = dist.is_initialized()

        if use_ddp:
            if self.config.ddp_type == 'native':
                model = NativeDDP(model,
                                  device_ids=[self.config.local_rank],
                                  bucket_cap_mb=100,
                                  gradient_as_bucket_view=self.config.
                                  use_gradient_as_bucket_view)
            elif self.config.ddp_type == 'apex':
                model = APEX_DDP(model,
                                 message_size=250000000,
                                 delay_allreduce=True,
                                 gradient_predivide_factor=torch.distributed.
                                 get_world_size())
            else:
                assert False, "Invalid DDP type"

        return model

    def on_backward(self,
                    step: int,
                    loss: torch.Tensor,
                    optimizer,
                    grad_scaler=None):
        if self.config.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if step % self.config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
