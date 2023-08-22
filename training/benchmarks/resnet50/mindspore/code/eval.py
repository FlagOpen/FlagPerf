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
"""eval resnet."""
import os
import sys
import time
import mindspore as ms
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.resnet import resnet50 as resnet
from src.dataset import create_dataset

# logger
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../../")))
from utils import flagperf_logger

ms.set_seed(1)


@moxing_wrapper()
def eval_net():
    """eval net"""
    target = config.device_target

    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    device_id = int(os.getenv('DEVICE_ID'))
    ms.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size, target=target)

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = ms.load_checkpoint(config.checkpoint_file_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction='mean',
                              smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    converged = 'true'
    if res['top_1_accuracy'] < config.target_acc1:
        converged = 'false'
    return converged, res['top_1_accuracy']

if __name__ == '__main__':
    run_logger = flagperf_logger.FlagPerfLogger()
    run_log_dir = os.path.join(os.getenv('RUN_LOG_DIR'), os.getenv('RANK_ID'))
    run_logger.init(run_log_dir, "flagperf_run.log", 'info', "both", log_caller=True)

    start_time = time.time()
    converged, acc = eval_net()

    finished_info = {
        "e2e_time": time.time() - start_time,
        "converged": converged,
        "acc": acc
    }
    run_logger.info("eval info:", finished_info)
