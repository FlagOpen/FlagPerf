# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train resnet."""
import os
import argparse
import ast
import moxing as mox
import mindspore as ms
import mindspore.nn as nn
import mindspore.log as logger

from mindspore.train.train_thor import ConvertModelUtils
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_group_size
from mindspore.parallel import set_algo_parameters

from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.eval_callback import EvalCallBack
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.resnet import resnet18 as resnet
from src.dataset import create_dataset1 as create_dataset
from modelarts.ResNet18.config import config


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--train_url', type=str, default='',
                    help='the path model saved')
parser.add_argument('--data_url', type=str, default='',
                    help='the training data')

parser.add_argument('--net', type=str, default="resnet18",
                    help='Resnet Model, resnet18, resnet34, '
                         'resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='Dataset, either cifar10 or imagenet2012')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False,
                    help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')

parser.add_argument('--dataset_path', type=str, default="/cache",
                    help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend',
                    choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None,
                    help='Pretrained checkpoint path')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False,
                    help='Run parameter server train')
parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                    help="Filter head weight parameters, default is False.")
parser.add_argument("--run_eval", type=ast.literal_eval, default=False,
                    help="Run evaluation when training, default is False.")
parser.add_argument('--eval_dataset_path', type=str, default=None,
                    help='Evaluation dataset path when run_eval is True')
parser.add_argument("--save_best_ckpt", type=ast.literal_eval, default=True,
                    help="Save best checkpoint when run_eval is True, "
                         "default is True.")
parser.add_argument("--eval_start_epoch", type=int, default=40,
                    help="Evaluation start epoch when run_eval is True, "
                         "default is 40.")
parser.add_argument("--eval_interval", type=int, default=1,
                    help="Evaluation interval when run_eval is True, "
                         "default is 1.")
parser.add_argument('--enable_cache', type=ast.literal_eval, default=False,
                    help='Caching the eval dataset in memory to speedup '
                         'evaluation, default is False.')
parser.add_argument('--cache_session_id', type=str, default="",
                    help='The session id for cache service.')
parser.add_argument('--mode', type=str, default='GRAPH',
                    choices=('GRAPH', 'PYNATIVE'),
                    help="Graph mode or PyNative mode, default is Graph mode")

parser.add_argument("--epoch_size", type=int, default=1,
                    help="training epoch size, default is 1.")
parser.add_argument("--num_classes", type=int, default=1001,
                    help="number of dataset categories, default is 1001.")


args_opt = parser.parse_args()

CKPT_OUTPUT_PATH = "./output"

ms.set_seed(1)

if config.optimizer == "Thor":
    if args_opt.device_target == "Ascend":
        from modelarts.ResNet18.config import config_thor_Ascend as config
    else:
        from modelarts.ResNet18.config import config_thor_gpu as config


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def set_graph_kernel_context(run_platform, net_name):
    if run_platform == "GPU" and net_name == "resnet101":
        ms.set_context(enable_graph_kernel=True,
                       graph_kernel_flags="--enable_parallel_fusion")


def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _export_air(ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return
    net = resnet(config.class_num)
    param_dict = ms.load_checkpoint(ckpt_file)
    ms.load_param_into_net(net, param_dict)

    input_arr = ms.numpy.zeros([1, 3, 304, 304], ms.float32)
    file_path = os.path.join(args_opt.train_url, "resnet")
    ms.export(net, input_arr, file_name=file_path, file_format="AIR")


def set_config():
    config.epoch_size = args_opt.epoch_size
    config.num_classes = args_opt.num_classes


def init_context(target):
    if args_opt.mode == 'GRAPH':
        ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
        set_graph_kernel_context(target, args_opt.net)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)

    if args_opt.parameter_server:
        ms.set_ps_context(enable_ps=True)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(
                device_num=args_opt.device_num,
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if args_opt.net == "resnet50" or args_opt.net == "se-resnet50":
                ms.set_auto_parallel_context(
                    all_reduce_fusion_config=[85, 160])
            elif args_opt.net == "resnet101":
                ms.set_auto_parallel_context(
                    all_reduce_fusion_config=[80, 210, 313])
            init()
        # GPU target
        else:
            init()
            ms.set_auto_parallel_context(
                device_num=get_group_size(),
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            if args_opt.net == "resnet50":
                ms.set_auto_parallel_context(
                    all_reduce_fusion_config=[85, 160])


def init_weight(net):
    if args_opt.pre_trained:
        param_dict = ms.load_checkpoint(args_opt.pre_trained)
        if args_opt.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        ms.load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    ms.common.initializer.initializer(ms.common.initializer.XavierUniform(),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))


def init_lr(step_size):
    if config.optimizer == "Thor":
        from src.lr_generator import get_thor_lr
        lr = get_thor_lr(0, config.lr_init, config.lr_decay,
                         config.lr_end_epoch, step_size, decay_epochs=39)
    else:
        if args_opt.net in ("resnet18", "resnet34", "resnet50", "se-resnet50"):
            lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end,
                        lr_max=config.lr_max,
                        warmup_epochs=config.warmup_epochs,
                        total_epochs=config.epoch_size,
                        steps_per_epoch=step_size,
                        lr_decay_mode=config.lr_decay_mode)
        else:
            lr = warmup_cosine_annealing_lr(
                config.lr, step_size, config.warmup_epochs, config.epoch_size,
                config.pretrain_epoch_size * step_size)
    return ms.Tensor(lr)


def define_opt(net, lr):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' \
                not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [
        {'params': decayed_params, 'weight_decay': config.weight_decay},
        {'params': no_decayed_params},
        {'order_params': net.trainable_params()}]
    opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)

    return opt


def define_model(net, opt, target):
    if args_opt.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    dist_eval_network = ClassifyCorrectCell(
        net) if args_opt.run_distribute else None
    metrics = {"acc"}
    if args_opt.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size,
                                       device_num=args_opt.device_num)}
    if (args_opt.net not in ("resnet18", "resnet50", "resnet101",
                             "se-resnet50")) or args_opt.parameter_server \
            or target == "CPU":
        # fp32 training
        model = ms.Model(net, loss_fn=loss, optimizer=opt, metrics=metrics,
                         eval_network=dist_eval_network)
    else:
        model = ms.Model(net, loss_fn=loss, optimizer=opt,
                         loss_scale_manager=loss_scale, metrics=metrics,
                         amp_level="O2", keep_batchnorm_fp32=False,
                         eval_network=dist_eval_network)
    return model, loss, loss_scale


def run_eval(model, target, ckpt_save_dir):
    if args_opt.eval_dataset_path is None \
            or (not os.path.isdir(args_opt.eval_dataset_path)):
        raise ValueError(
            "{} is not a existing path.".format(args_opt.eval_dataset_path))
    eval_dataset = create_dataset(
        dataset_path=args_opt.eval_dataset_path,
        do_train=False,
        batch_size=config.batch_size,
        target=target,
        enable_cache=args_opt.enable_cache,
        cache_session_id=args_opt.cache_session_id)
    eval_param_dict = {"model": model, "dataset": eval_dataset,
                       "metrics_name": "acc"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict,
                           interval=args_opt.eval_interval,
                           eval_start_epoch=args_opt.eval_start_epoch,
                           save_best_ckpt=args_opt.save_best_ckpt,
                           ckpt_directory=ckpt_save_dir,
                           besk_ckpt_name="best_acc.ckpt",
                           metrics_name="acc")
    return eval_cb


def main():
    set_config()
    target = args_opt.device_target
    if target == "CPU":
        args_opt.run_distribute = False

    # init context
    init_context(target)
    ckpt_save_dir = CKPT_OUTPUT_PATH

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True,
                             batch_size=config.batch_size, target=target,
                             distribute=args_opt.run_distribute)

    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)
    if args_opt.parameter_server:
        net.set_param_ps()

    # init weight
    init_weight(net)

    # init lr
    lr = init_lr(step_size)

    # define opt
    opt = define_opt(net, lr)

    # define model
    model, loss, loss_scale = define_model(net, opt, target)

    if config.optimizer == "Thor" and args_opt.dataset == "imagenet2012":
        from src.lr_generator import get_thor_damping
        damping = get_thor_damping(0, config.damping_init, config.damping_decay,
                                   70, step_size)
        split_indices = [26, 53]
        opt = nn.thor(net, lr, ms.Tensor(damping), config.momentum,
                      config.weight_decay, config.loss_scale,
                      config.batch_size, split_indices=split_indices,
                      frequency=config.frequency)
        model = ConvertModelUtils().convert_to_thor_model(
            model=model, network=net, loss_fn=loss, optimizer=opt,
            loss_scale_manager=loss_scale, metrics={'acc'}, amp_level="O2",
            keep_batchnorm_fp32=False)
        args_opt.run_eval = False
        logger.warning("Thor optimizer not support evaluation while training.")

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir,
                                  config=config_ck)
        cb += [ckpt_cb]
    if args_opt.run_eval:
        eval_cb = run_eval(model, target, ckpt_save_dir)
        cb += [eval_cb]
    # train model
    if args_opt.net == "se-resnet50":
        config.epoch_size = config.train_epoch_size
    dataset_sink_mode = (not args_opt.parameter_server) and target != "CPU"
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset,
                callbacks=cb,
                sink_size=dataset.get_dataset_size(),
                dataset_sink_mode=dataset_sink_mode)

    if args_opt.run_eval and args_opt.enable_cache:
        print("Remember to shut down the cache server via \"cache_admin --stop\"")


if __name__ == '__main__':

    mox.file.copy_parallel(args_opt.data_url, '/cache')
    main()
    if not os.path.exists(CKPT_OUTPUT_PATH):
        os.makedirs(CKPT_OUTPUT_PATH)
    _export_air(CKPT_OUTPUT_PATH)
    mox.file.copy_parallel(CKPT_OUTPUT_PATH, args_opt.train_url)
