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
# sys
import datetime
import glob
import os
import numpy as np
import time
import sys
import time

# logger
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../../")))
from utils import flagperf_logger

# mindspore
import mindspore as ms
import mindspore.nn as nn
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters
import mindspore.log as logger

# resnet code
from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.eval_callback import EvalCallBack
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_rank_id, get_device_num
from src.resnet import conv_variance_scaling_initializer
from src.resnet import resnet50 as resnet
from src.dataset import create_dataset

ms.set_seed(1)


class LossCallBack(LossMonitor):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            # pylint: disable=line-too-long
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss), flush=True)


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


def set_parameter():
    """set_parameter"""
    target = config.device_target

    # init context
    rank_save_graphs_path = os.path.join(config.save_graphs_path, "soma", str(os.getenv('DEVICE_ID')))
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=config.save_graphs,
                           save_graphs_path=rank_save_graphs_path)

    if config.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        ms.set_context(device_id=device_id)
        ms.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        if config.net_name == "resnet50" or config.net_name == "se-resnet50":
            if config.boost_mode not in ["O1", "O2"]:
                ms.set_auto_parallel_context(
                    all_reduce_fusion_config=config.all_reduce_fusion_config)
        elif config.net_name in ["resnet101", "resnet152"]:
            ms.set_auto_parallel_context(
                all_reduce_fusion_config=config.all_reduce_fusion_config)
        init()


def load_pre_trained_checkpoint():
    """
    Load checkpoint according to pre_trained path.
    """
    param_dict = None
    if config.pre_trained:
        if os.path.isdir(config.pre_trained):
            ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path, "ckpt_0")
            ckpt_pattern = os.path.join(ckpt_save_dir, "*.ckpt")
            ckpt_files = glob.glob(ckpt_pattern)
            if not ckpt_files:
                logger.warning(f"There is no ckpt file in {ckpt_save_dir}, "
                               f"pre_trained is unsupported.")
            else:
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                time_stamp = datetime.datetime.now()
                print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')}"
                      f" pre trained ckpt model {ckpt_files[0]} loading",
                      flush=True)
                param_dict = ms.load_checkpoint(ckpt_files[0])
        elif os.path.isfile(config.pre_trained):
            param_dict = ms.load_checkpoint(config.pre_trained)
        else:
            print(f"Invalid pre_trained {config.pre_trained} parameter.")
    return param_dict


def init_weight(net, param_dict):
    """init_weight"""
    if config.pre_trained:
        if param_dict:
            if param_dict.get("epoch_num") and param_dict.get("step_num"):
                config.has_trained_epoch = int(param_dict["epoch_num"].data.asnumpy())
                config.has_trained_step = int(param_dict["step_num"].data.asnumpy())
            else:
                config.has_trained_epoch = 0
                config.has_trained_step = 0

            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(param_dict, filter_list)
            ms.load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if config.conv_init == "XavierUniform":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.XavierUniform(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif config.conv_init == "TruncatedNormal":
                    weight = conv_variance_scaling_initializer(cell.in_channels,
                                                               cell.out_channels,
                                                               cell.kernel_size[0])
                    cell.weight.set_data(weight)
            if isinstance(cell, nn.Dense):
                if config.dense_init == "TruncatedNormal":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif config.dense_init == "RandomNormal":
                    in_channel = cell.in_channels
                    out_channel = cell.out_channels
                    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                    weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                    cell.weight.set_data(weight)


def init_lr(step_size):
    """init lr"""
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    return lr


def init_loss_scale():
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    return loss


def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


def run_eval(target, model, ckpt_save_dir, cb):
    """run_eval"""
    if config.run_eval:
        if config.eval_dataset_path is None or (not os.path.isdir(config.eval_dataset_path)):
            raise ValueError("{} is not a existing path.".format(config.eval_dataset_path))
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False,
                                      batch_size=config.batch_size, train_image_size=config.train_image_size,
                                      eval_image_size=config.eval_image_size,
                                      target=target, enable_cache=config.enable_cache,
                                      cache_session_id=config.cache_session_id)
        eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=config.save_best_ckpt,
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="acc")
        cb += [eval_cb]


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    if config.enable_modelarts and config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank_id()) + "/"
    else:
        if config.run_distribute:
            ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


@moxing_wrapper()
def train_net():
    """train net"""
    target = config.device_target
    set_parameter()
    ckpt_param_dict = load_pre_trained_checkpoint()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)

    init_weight(net=net, param_dict=ckpt_param_dict)
    lr = ms.Tensor(init_lr(step_size=step_size))
    # define opt
    group_params = init_group_params(net)
    opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    if config.optimizer == "LARS":
        opt = nn.LARS(opt, epsilon=config.lars_epsilon, coefficient=config.lars_coefficient,
                      lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)
    loss = init_loss_scale()
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    dist_eval_network = ClassifyCorrectCell(net) if config.run_distribute else None
    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                     amp_level="O2", keep_batchnorm_fp32=False, eval_network=dist_eval_network)

    # build
    build_start_time = time.time()
    model.build(dataset, None, sink_size=dataset.get_dataset_size(), epoch=config.epoch_size - config.pretrain_epoch_size)
    build_time = time.time() - build_start_time

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(config.has_trained_epoch)
    cb = [time_cb, loss_cb]
    ckpt_save_dir = set_save_ckpt_dir()
    if config.save_checkpoint:
        ckpt_append_info = [{"epoch_num": config.has_trained_epoch, "step_num": config.has_trained_step}]
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max,
                                     append_info=ckpt_append_info)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    run_eval(target, model, ckpt_save_dir, cb)

    # train model
    train_start_time = time.time()
    real_epoch = config.epoch_size - config.has_trained_epoch
    model.train(real_epoch, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)
    train_time = time.time() - train_start_time

    if config.run_eval and config.enable_cache:
        print("Remember to shut down the cache server via \"cache_admin --stop\"")

    return build_time, train_time, int(os.getenv('RANK_SIZE')) * step_size * config.batch_size * real_epoch

if __name__ == '__main__':
    run_logger = flagperf_logger.FlagPerfLogger()
    run_log_dir = os.path.join(os.getenv('LOG_PATH'), os.getenv('RANK_ID'))
    run_logger.init(run_log_dir, "flagperf_run.log", 'info', "both", log_caller=True)

    start_time = time.time()
    build_time, train_time, total_samples = train_net()

    finished_info = {
        "e2e_time": time.time() - start_time,
        "train_time": train_time,
        "pure_training_computing_time": train_time - build_time,
        "throughput(ips)_raw": total_samples / train_time,
        "throughput(ips)_pure_compute": total_samples / (train_time - build_time),
    }
    run_logger.info("--------------Train Info--------------------")
    run_logger.info("e2e_time:" + str(time.time() - start_time))
    run_logger.info("train_time:" + str(train_time))
    run_logger.info("pure_training_computing_time:" + str(train_time - build_time))
    run_logger.info("throughput(ips)_raw:" + str(total_samples / train_time))
    run_logger.info("throughput(ips)_pure_compute:" + str(total_samples / (train_time - build_time)))

