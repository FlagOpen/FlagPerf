import torch
import torch_musa

import config


def create_grad_scaler():
    """create_grad_scaler for mixed precision training"""
    scaler = torch_musa.amp.GradScaler() if config.amp else None
    return scaler


def train_one_step(model, batch_data, optimizer, cur_step, scaler=None):
    input_ids, labels = batch_data
    if scaler:
        with torch_musa.amp.autocast(enabled=True):
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

        scaler.scale(loss).backward()
        if cur_step % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
    else:
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        if cur_step % config.gradient_accumulation_steps == 0:
            optimizer.step()

    return loss
    