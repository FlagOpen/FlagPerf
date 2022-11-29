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

    data_dir = get_config_arg(config, "data_dir")
    config.data_dir = data_dir
    if data_dir is None:
        raise ValueError("Invalid data_dir, should be given a path.")
    if not ospath.isdir(data_dir):
        raise ValueError(f"data_dir '{data_dir}' not exists.")

    train_data = get_config_arg(config, "train_data")
    if not train_data:
        train_data = ospath.join(
            data_dir,
            'ReCoRD/glm_train_eval_hdf5_sparse/train_hdf5/train_sparse.hdf5')
    if not ospath.isfile(train_data):
        raise ValueError(f"train_data '{train_data}' not exists.")
    config.train_data = train_data

    eval_data = get_config_arg(config, "eval_data")
    if not eval_data:
        eval_data = ospath.join(
            data_dir,
            'ReCoRD/glm_train_eval_hdf5_sparse/eval_hdf5/eval_sparse.hdf5')
    if not ospath.isfile(eval_data):
        raise ValueError(f"eval_data '{eval_data}' not exists.")
    config.eval_data = eval_data

    init_checkpoint = get_config_arg(config, "init_checkpoint")
    if not init_checkpoint:
        init_checkpoint = ospath.join(
            data_dir, 'blocklm-large-blank/200000/mp_rank_00_model_states.pt')
    if not ospath.exists(init_checkpoint):
        raise ValueError(f"init_checkpoint '{init_checkpoint}' not exists.")
    config.init_checkpoint = init_checkpoint

    if config.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(config.gradient_accumulation_steps))

    one_step_train_samples = config.n_gpu * config.train_batch_size * config.gradient_accumulation_steps
    print('config.n_gpu', config.n_gpu)
    print('config.train_batch_size', config.train_batch_size)
    print('config.gradient_accumulation_steps',
          config.gradient_accumulation_steps)
    config.train_iters = config.max_samples_termination / one_step_train_samples

    if config.eval_interval_samples == 0:
        eval_interval_samples = (
            0.05 * (230.23 * global_batch_size(config) + 3000000)) / 25000
        eval_interval_samples = int(eval_interval_samples) * 25000
        config.eval_interval_samples = eval_interval_samples
