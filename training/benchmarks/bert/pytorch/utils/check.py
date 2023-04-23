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
    print(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(config.device, config.n_gpu, config.local_rank != -1,
               config.fp16))

    train_dir = get_config_arg(config, "train_dir")

    data_dir = get_config_arg(config, "data_dir")
    config.data_dir = data_dir
    if data_dir is None and train_dir is None:
        raise ValueError(
            "Invalid data_dir and train_dir, should be given a path.")

    if train_dir is None:
        config.train_dir = ospath.join(data_dir, "2048_shards_uncompressed")

    init_checkpoint = get_config_arg(config, "init_checkpoint")
    config.init_checkpoint = init_checkpoint
    if init_checkpoint is None:
        if data_dir is None:
            raise ValueError(
                "Invalid init_checkpoint and data_dir, should be given a path."
            )
        config.init_checkpoint = ospath.join(data_dir, "model.ckpt-28252.pt")

    bert_config_path = get_config_arg(config, "bert_config_path")
    config.bert_config_path = bert_config_path
    if bert_config_path is None:
        if data_dir is None:
            raise ValueError(
                "Invalid bert_config_path and data_dir, should be given a path."
            )
        config.bert_config_path = ospath.join(data_dir, "bert_config.json")

    eval_dir = get_config_arg(config, "eval_dir")
    config.eval_dir = eval_dir
    if eval_dir is None:
        if data_dir is None:
            raise ValueError(
                "Invalid eval_dir and data_dir, should be given a path.")
        config.eval_dir = ospath.join(data_dir, "eval_set_uncompressed")

    if config.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(config.gradient_accumulation_steps))
    # if args.train_batch_size % args.gradient_accumulation_steps != 0:
    #     raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
    #         args.gradient_accumulation_steps, args.train_batch_size))
    #
    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if config.eval_interval_samples == 0:
        eval_interval_samples = (
            0.05 * (230.23 * global_batch_size(config) + 3000000)) / 25000
        eval_interval_samples = int(eval_interval_samples) * 25000
        config.eval_interval_samples = eval_interval_samples

    # if not (config.do_train or (config.eval_dir and config.eval_iter_samples <= 0)):
    #     raise ValueError(" `do_train`  or should be in offline eval mode")

    # if not config.resume_from_checkpoint or not os.path.exists(config.output_dir):
    #     os.makedirs(config.output_dir, exist_ok=True)
