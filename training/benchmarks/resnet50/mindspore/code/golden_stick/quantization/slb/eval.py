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
"""eval resnet."""

import mindspore as ms
import mindspore.log as logger
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from slb import create_slb
from src.resnet import resnet18 as resnet
from src.model_utils.config import config

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset

ms.set_seed(1)

def eval_net():
    """eval net"""
    target = config.device_target
    if target != "GPU":
        logger.warning("SLB only support GPU now!")

    # init context
    if config.mode_name == "GRAPH":
        ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size, target=target)

    # define net
    net = resnet(class_num=config.class_num)
    algo = create_slb(config.quant_type)
    net = algo.apply(net)

    # load checkpoint
    param_dict = ms.load_checkpoint(config.checkpoint_file_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", config.checkpoint_file_path)


if __name__ == '__main__':
    eval_net()
