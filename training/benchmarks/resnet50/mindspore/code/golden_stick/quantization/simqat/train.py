# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from simqat import create_simqat
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.metric import DistAccuracy
from src.resnet import conv_variance_scaling_initializer
from src.resnet import resnet50 as resnet
from src.model_utils.config import config

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    if config.mode_name == "GRAPH":
        from src.dataset import create_dataset2 as create_dataset
    else:
        from src.dataset import create_dataset_pynative as create_dataset

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


def set_parameter():
    """set_parameter"""
    target = config.device_target
    if target == "CPU":
        config.run_distribute = False

    # init context
    if config.mode_name == "GRAPH":
        ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)

    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if config.boost_mode not in ["O1", "O2"]:
                ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            init()
        # GPU target
        else:
            init()
            ms.set_auto_parallel_context(device_num=config.device_num,
                                         parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)


def init_weight(net):
    """init_weight"""
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


def load_fp32_ckpt(net):
    if config.fp32_ckpt:
        if os.path.isfile(config.fp32_ckpt):
            ckpt = ms.load_checkpoint(config.fp32_ckpt)
            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(ckpt, filter_list)
            ms.load_param_into_net(net, ckpt)
        else:
            print(f"Invalid fp32_ckpt {config.fp32_ckpt} parameter.")


def load_pretrained_ckpt(net):
    if config.pre_trained:
        if os.path.isfile(config.pre_trained):
            ckpt = ms.load_checkpoint(config.pre_trained)
            if ckpt.get("epoch_num") and ckpt.get("step_num"):
                config.has_trained_epoch = int(ckpt["epoch_num"].data.asnumpy())
                config.has_trained_step = int(ckpt["step_num"].data.asnumpy())
            else:
                config.has_trained_epoch = 0
                config.has_trained_step = 0

            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(ckpt, filter_list)
            ms.load_param_into_net(net, ckpt)
        else:
            print(f"Invalid pre_trained {config.pre_trained} parameter.")


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


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    if config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


def init_loss_scale():
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


def train_net():
    """train net"""
    print("Train configure: {}".format(config))
    target = config.device_target
    if target != "GPU":
        raise NotImplementedError("SimQAT only support running on GPU now!")
    set_parameter()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)

    init_weight(net)
    load_fp32_ckpt(net)
    algo = create_simqat()
    net = algo.apply(net)
    load_pretrained_ckpt(net)

    lr = get_lr(lr_init=config.lr_init,
                lr_end=0.0,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode='cosine')
    if config.pre_trained:
        lr = lr[config.has_trained_epoch * step_size:]
    lr = ms.Tensor(lr)
    # define opt
    group_params = init_group_params(net)
    opt = nn.Momentum(group_params, lr, config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    loss = init_loss_scale()
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                     amp_level="O0", boost_level=config.boost_mode, keep_batchnorm_fp32=False,
                     boost_config_dict={"grad_freeze": {"total_steps": config.epoch_size * step_size}})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(config.has_trained_epoch)
    cb = [time_cb, loss_cb]
    if algo:
        algo_cb = algo.callback()
        cb.append(algo_cb)
    ckpt_save_dir = set_save_ckpt_dir()
    if config.save_checkpoint:
        ckpt_append_info = [{"epoch_num": config.has_trained_epoch, "step_num": config.has_trained_step}]
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max,
                                     append_info=ckpt_append_info)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    # train model
    dataset_sink_mode = target != "CPU"
    model.train(config.epoch_size - config.has_trained_epoch, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    train_net()
