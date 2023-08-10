import torch


def torch_sync(config):
    if config.vendor == "nvidia":
        torch.cuda.synchronize()
    if config.vendor == "kunlunxin":
        # kunlunxin case
        # xpu sync already finsh after InferModel.__call__
        pass
