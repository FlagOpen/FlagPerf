from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync
import torch_tensorrt


def cal_perf(config, dataloader_len, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * dataloader_len * config.batch_size / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " sps")
    model_forward_core_perf = config.repeat * dataloader_len * config.batch_size / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " sps")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def model_forward(model, dataloader, evaluator, config):
    start = time.time()

    core_time = 0.0

    correct = 1
    whole = 1

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))
        for step, (x, y) in enumerate(dataloader):
            torch_sync(config)
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)) + " MaskedLM acc:" +
                             str(round(correct / whole, 3)))

            with torch.no_grad():

                x = x.int().cuda()
                y = y.int().cuda()
                pred = model(x)
                torch_sync(config)
                core_time += time.time() - core_time_start

                correct_iter, whole_iter = evaluator(pred, x, y)

                correct += correct_iter
                whole += whole_iter

    acc = correct / whole

    logger.info("MaskedLM Acc: " + str(acc))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(acc, 3)


def engine_forward(model, dataloader, evaluator, config):
    dummy_input = torch.ones(config.batch_size, config.seq_length).long().cuda()
    traced_model = torch.jit.trace(model, [dummy_input])
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(dummy_input.shape, dtype=torch.int32)],
        enabled_precisions={torch.float32, torch.float16},
        truncate_long_and_double=True)

    start = time.time()

    core_time = 0.0

    correct = 1
    whole = 1

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))
        for step, (x, y) in enumerate(dataloader):
            torch_sync(config)
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)) + " MaskedLM acc:" +
                             str(round(correct / whole, 3)))

            with torch.no_grad():

                x = x.int().cuda()
                y = y.int().cuda()
                pred = trt_model(x)
                torch_sync(config)
                core_time += time.time() - core_time_start

                correct_iter, whole_iter = evaluator(pred, x, y)

                correct += correct_iter
                whole += whole_iter

    acc = correct / whole

    logger.info("MaskedLM Acc: " + str(acc))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(acc, 3)
