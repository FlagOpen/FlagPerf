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
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore_gs import PrunerKfCompressAlgo, PrunerFtCompressAlgo
from mindspore_gs.pruner.scop.scop_pruner import KfConv2d, MaskedConv2dbn
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
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


class NetWithLossCell(nn.WithLossCell):
    """Calculate NetWithLossCell."""

    def __init__(self, backbone, loss_fn, ngpu):
        super(NetWithLossCell, self).__init__(backbone, loss_fn)
        self.ngpu = ngpu

    def construct(self, data, label):
        num_pgpu = data.shape[0] // 2 * self.ngpu
        out = self._backbone(data)
        output_list = []
        for igpu in range(self.ngpu):
            output_list.append(out[igpu * num_pgpu * 2:igpu * num_pgpu * 2 + num_pgpu])
        out = ops.Concat(axis=0)(output_list)
        return self._loss_fn(out, label)


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
        else:
            # GPU target
            init()
            ms.set_auto_parallel_context(device_num=config.device_num,
                                         parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)


def init_weight(net, param_dict):
    """init_weight"""
    if config.pre_trained and param_dict:
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
    set_parameter()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)

    # apply golden-stick algo
    algo_kf = PrunerKfCompressAlgo({})
    pre_ckpt = ms.load_checkpoint(config.fp32_ckpt)
    ms.load_param_into_net(net, pre_ckpt)
    model = algo_kf.apply(net)

    kfconv_list = []
    for _, (_, module) in enumerate(model.cells_and_names()):
        if isinstance(module, KfConv2d):
            kfconv_list.append(module)
    kfscale_list = [[] for _ in range(len(kfconv_list))]

    for param in model.get_parameters():
        param.requires_grad = False
    for _, (_, module) in enumerate(model.cells_and_names()):
        if isinstance(module, KfConv2d):
            module.kfscale.requires_grad = True

    lr = get_lr(lr_init=config.lr_init,
                lr_end=0.0,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode='cosine')

    optimizer = nn.Momentum(filter(lambda p: p.requires_grad, model.get_parameters()),
                            learning_rate=lr,
                            momentum=config.momentum,
                            loss_scale=config.loss_scale
                            )

    kf_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_with_loss = NetWithLossCell(model, kf_loss_fn, 1)

    net_train_step = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_kf(dataset, net_train_step, model, kfconv_list, kfscale_list)


def train_kf(dataset, net_train_step, model, kfconv_list, kfscale_list):
    """train konckoff."""
    for _ in range(0, config.epoch_size):
        from copy import deepcopy
        for _, (kf_data, kf_target) in enumerate(dataset.create_tuple_iterator()):
            kf = deepcopy(kf_data)
            idx = ops.Randperm(max_length=kf.shape[0])(ms.Tensor([kf.shape[0]], dtype=mstype.int32))
            kf_input = kf[idx, :].view(kf.shape)

            input_list = []
            kf_num_pgpu = kf_data.shape[0] // config.ngpu
            for i_gpu in range(config.ngpu):
                input_list.append(ops.Concat(axis=0)(
                    [kf_data[i_gpu * kf_num_pgpu:(i_gpu + 1) * kf_num_pgpu],
                     kf_input[i_gpu * kf_num_pgpu:(i_gpu + 1) * kf_num_pgpu]]))
            kf_input = ops.Concat(axis=0)(input_list)
            loss = net_train_step(kf_input, kf_target)
            print('loss:{}'.format(loss))

            for module in model.cells():
                if isinstance(module, KfConv2d):
                    module.kfscale = module.kfscale._update_tensor_data(
                        ops.clip_by_value(module.kfscale.data, clip_value_max=ms.Tensor(1, mstype.float32),
                                          clip_value_min=ms.Tensor(0, mstype.float32)))
        for ikf in range(len(kfconv_list)):
            kfscale_list[ikf].append(kfconv_list[ikf].kfscale.data.clone())

    for param in model.get_parameters():
        param.requires_grad = True
    for kfscale in kfscale_list[10]:
        print(ops.Squeeze()(kfscale).asnumpy())
    for kfscale_last in kfscale_list:
        print(ops.Squeeze()(kfscale_last[-1]).asnumpy())

    for _, (_, module) in enumerate(model.cells_and_names()):
        if isinstance(module, KfConv2d):
            module.score = module.bn.gamma.data.abs() * ops.Squeeze()(module.kfscale.data - (1 - module.kfscale.data))

    for kfconv in kfconv_list:
        kfconv.prune_rate = config.prune_rate
    for _, (_, module) in enumerate(model.cells_and_names()):
        if isinstance(module, KfConv2d):
            _, index = ops.Sort()(module.score)
            num_pruned_channel = int(module.prune_rate * module.score.shape[0])
            module.out_index = index[num_pruned_channel:]
    train_ft(model, dataset)


def train_ft(model, dataset):
    """train finetune."""
    algo_ft = PrunerFtCompressAlgo({})
    model = algo_ft.apply(model)
    lr_ft_new = ms.Tensor(get_lr(lr_init=config.lr_init,
                                 lr_end=config.lr_ft_end,
                                 lr_max=config.lr_ft_max,
                                 warmup_epochs=config.warmup_epochs,
                                 total_epochs=config.epochs_ft,
                                 steps_per_epoch=dataset.get_dataset_size(),
                                 lr_decay_mode='poly'))

    optimizer_ft = nn.Momentum(filter(lambda p: p.requires_grad, model.get_parameters()),
                               learning_rate=lr_ft_new,
                               momentum=config.momentum,
                               loss_scale=config.loss_scale
                               )
    model.set_train()

    metrics = {"acc"}
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    ft_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model_ft = ms.Model(model, loss_fn=ft_loss_fn, optimizer=optimizer_ft, loss_scale_manager=loss_scale,
                        metrics=metrics,
                        amp_level="O2", boost_level="O0", keep_batchnorm_fp32=False)

    step_size = dataset.get_dataset_size()

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=5 * step_size,
                                 keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory=config.output_path,
                              config=config_ck)
    ft_cb = [time_cb, loss_cb, ckpt_cb]

    model_ft.train(config.epochs_ft, dataset, callbacks=ft_cb,
                   sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)

    masked_conv_list = []
    for _, (nam, module) in enumerate(model.cells_and_names()):
        if isinstance(module, MaskedConv2dbn):
            masked_conv_list.append((nam, module))
    for imd in range(len(masked_conv_list)):
        if 'conv2' in masked_conv_list[imd][0] or 'conv3' in masked_conv_list[imd][0]:
            masked_conv_list[imd][1].in_index = masked_conv_list[imd - 1][1].out_index


if __name__ == '__main__':
    train_net()
