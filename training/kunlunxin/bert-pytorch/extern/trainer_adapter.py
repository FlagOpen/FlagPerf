import os
from typing import Tuple

import torch
import torch.distributed as dist

from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim import Optimizer
from torch_xmlir.optimizer import Lamb

import torch_xmlir.core.xpu_model as xm

import utils
import config

BERT_MODEL = torch.nn.Module


def convert_model(model: BERT_MODEL) -> BERT_MODEL:
    return model


def create_optimizer(model: BERT_MODEL) -> Optimizer:
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        config.weight_decay_rate
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    if config.distributed_lamb:
        print("distributed")
        #DistributedFusedLAMB._pipeline_block_reductions = _pipeline_block_reductions_patched
        #DistributedFusedLAMB._pipeline_step = _pipeline_step_patched
        #optimizer = DistributedFusedLAMB(optimizer_grouped_parameters, lr=config.learning_rate,
        #                                 betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_2),
        #                                 eps=1e-6,
        #                                 max_grad_norm=1.0,
        #                                 overlap_reductions=config.dwu_overlap_reductions,
        #                                 dwu_group_size=config.dwu_group_size,
        #                                 dwu_num_blocks=config.dwu_num_blocks,
        #                                 dwu_num_chunks=config.dwu_num_chunks,
        #                                 dwu_num_rs_pg=config.dwu_num_rs_pg,
        #                                 dwu_num_ar_pg=config.dwu_num_ar_pg,
        #                                 dwu_num_ag_pg=config.dwu_num_ag_pg,
        #                                 use_nvlamb=False,
        #                                 e5m2_allgather=config.dwu_e5m2_allgather)
        #optimizer.set_global_scale(float(os.getenv("INIT_LOSS_SCALE", 2 ** 20)))
    else:
        optimizer = Lamb(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         betas=(config.opt_lamb_beta_1,
                                config.opt_lamb_beta_2))

    return optimizer


def model_to_fp16(model: BERT_MODEL,
                  optimizer: Optimizer) -> Tuple[BERT_MODEL, Optimizer]:
    return model, optimizer


def model_to_ddp(model: BERT_MODEL) -> BERT_MODEL:
    use_ddp = dist.is_initialized()
    if use_ddp and config.use_xpu:
        from torch_xmlir.distributed import DistributedDataParallel as DDP
        model = DDP(model)
    return model


def backward(step: int,
             loss: torch.Tensor,
             optimizer: Optimizer,
             grad_scaler: GradScaler = None):
    if config.bypass_amp:
        loss.backward()
    elif config.distributed_lamb:
        optimizer._lazy_init_stage1()
        grad_scaler.scale(loss).backward()
        optimizer._lazy_init_stage2()
    else:
        with amp.scale_loss(loss,
                            optimizer,
                            delay_overflow_check=self.config.
                            allreduce_post_accumulation) as scaled_loss:
            scaled_loss.backward()

    update_step = step % config.gradient_accumulation_steps == 0
    if update_step:
        update_model_params(loss, optimizer, grad_scaler)
    else:
        xm.mark_step()


def update_model_params(loss,
                        optimizer: Optimizer,
                        grad_scaler: GradScaler = None):
    if config.allreduce_post_accumulation and config.use_cuda_graph:
        assert False, "code path not tested with cuda graphs"
    if config.distributed_lamb:
        optimizer.set_global_scale(grad_scaler._get_scale_async())
        optimizer.complete_reductions()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        found_inf = optimizer._overflow_buf  # GPU tensor

    elif config.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        # torch.nn.utils.clip_grad_norm_(parameters=amp.master_params(optimizer), max_norm=1.0, norm_type=2.0)
        #scaler = _amp_state.loss_scalers[0]
        # master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        # flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if config.allreduce_post_accumulation_fp16 else torch.float32
        # flat_raw = torch.empty(flat_grad_size, device='cpu', dtype=allreduce_dtype)
        #flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        #allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        #self.overflow_buf.zero_()
        #amp_C.multi_tensor_scale(65536,
        #                         self.overflow_buf,
        #                         [master_grads, allreduced_views],
        #                         scaler.loss_scale() / (
        #                                 torch.distributed.get_world_size() * config.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        #torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        #self.overflow_buf.zero_()
        #amp_C.multi_tensor_scale(65536,
        #                         self.overflow_buf,
        #                         [allreduced_views, master_grads],
        #                         1. / scaler.loss_scale())
        # 5. update loss scale
        had_overflow = 0
        #scaler = _amp_state.loss_scalers[0]
        #old_overflow_buf = scaler._overflow_buf
        #scaler._overflow_buf = self.overflow_buf
        #had_overflow = scaler.update_scale()
        #scaler._overflow_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            # optimizer.step()
            xm.optimizer_step(optimizer, barrier=True)
        else:
            # Overflow detected, print message and clear gradients
            if utils.is_main_process():
                print("Overflow detected, reduced loss_scaler to %f" %
                      (scaler.loss_scale()))
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
    else:
        xm.optimizer_step(optimizer, barrier=True)

    optimizer.zero_grad()
