from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim import Optimizer
from torch.optim import Adam
import torch
import config
from typing import Tuple

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

    optimizer = Adam(optimizer_grouped_parameters,
                     lr=config.learning_rate,
                     betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_2))

    return optimizer


def model_to_fp16(model: BERT_MODEL,
                  optimizer: Optimizer) -> Tuple[BERT_MODEL, Optimizer]:
    return model, optimizer


def model_to_ddp(model: BERT_MODEL) -> BERT_MODEL:
    return model


def backward(step: int,
             loss: torch.Tensor,
             optimizer: Optimizer,
             grad_scaler: GradScaler = None):
    loss.backward()

    update_step = step % config.gradient_accumulation_steps == 0
    if update_step:
        update_model_params(loss, optimizer, grad_scaler)


def update_model_params(loss,
                        optimizer: Optimizer,
                        grad_scaler: GradScaler = None):
    optimizer.step()
    optimizer.zero_grad()
