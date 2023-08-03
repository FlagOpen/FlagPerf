from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync


def cal_perf(config, dataloader_len, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * dataloader_len * config.batch_size / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " qps")
    model_forward_core_perf = config.repeat * dataloader_len * config.batch_size / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " qps")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def model_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    start = time.time()
    core_time = 0.0

    correct = 1
    whole = 1

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

                pred = pred[0]
                pred = torch.argmax(pred, dim=2)
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
    start = time.time()
    core_time = 0.0
    foo_time = 0.0

    correct = 1
    whole = 1

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

                pred = pred[0]
                pred = pred.reshape(config.batch_size, config.seq_length, -1)
                pred = torch.argmax(pred, dim=2)
                pred = pred.cpu()
                correct_iter, whole_iter = evaluator(pred, x, y)

                correct += correct_iter
                whole += whole_iter

    acc = correct / whole

    logger.info("MaskedLM Acc: " + str(acc))

    duration = time.time() - start - foo_time
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time - foo_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(acc, 3)
