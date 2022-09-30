import os
import os.path as ospath
from .dist import global_batch_size


def get_config_arg(config, name):
    if hasattr(config, name):
        value = getattr(config, name)
        if value is not None:
            return value

    if name in os.environ:
        return os.environ[name]

    return None


def check_config(config):
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        config.device, config.n_gpu, config.local_rank != -1, config.fp16))

    data_dir = get_config_arg(config, "data_dir")

    init_checkpoint = get_config_arg(config, "init_checkpoint")
    config.init_checkpoint = init_checkpoint
    if init_checkpoint is None:
        if data_dir is None:
            raise ValueError("Invalid init_checkpoint and data_dir, should be given a path.")
        config.init_checkpoint = ospath.join(data_dir, "cpm_model_states_medium.pt")

    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config.gradient_accumulation_steps))

