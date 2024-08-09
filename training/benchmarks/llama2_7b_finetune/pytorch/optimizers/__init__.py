import torch.optim as optim


def create_optimizer(model, train_config):
    opt = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        fused=True if train_config.use_fp16 else False
    )
    return opt
