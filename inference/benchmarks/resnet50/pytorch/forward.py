# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync


def cal_perf(config, dataloader_len, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * dataloader_len * config.batch_size / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " ips")
    model_forward_core_perf = config.repeat * dataloader_len * config.batch_size / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " ips")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def model_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    start = time.time()
    core_time = 0.0
    acc = []

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        all_top1 = []
        for step, (x, y) in enumerate(dataloader):
            torch_sync(config)
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))

            with torch.no_grad():

                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                torch_sync(config)
                core_time += time.time() - core_time_start

                top1 = evaluator(pred, y)

                all_top1.extend(top1.cpu())

        acc.append(np.mean(all_top1))

    logger.info("Top1 Acc: " + str(acc))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(acc)), 3)


def engine_forward(model, dataloader, evaluator, config):
    start = time.time()
    core_time = 0.0
    foo_time = 0.0
    acc = []

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        all_top1 = []
        for step, (x, y) in enumerate(dataloader):
            torch_sync(config)
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))

            with torch.no_grad():

                outputs = model([x])
                pred = outputs[0]
                foo_time += outputs[1]

                torch_sync(config)
                core_time += time.time() - core_time_start

                pred = pred[0].float()
                pred = pred.reshape(config.batch_size, -1)
                pred = pred.cpu()
                top1 = evaluator(pred, y)

                all_top1.extend(top1.cpu())

        acc.append(np.mean(all_top1))

    logger.info("Top1 Acc: " + str(acc))

    duration = time.time() - start - foo_time
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time - foo_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(acc)), 3)
