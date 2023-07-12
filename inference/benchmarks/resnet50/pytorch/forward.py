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

                top1 = evaluator(pred, y)

                all_top1.extend(top1.cpu())
            core_time += time.time() - core_time_start

        acc.append(np.mean(all_top1))

    logger.info("Top1 Acc: " + str(acc))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(acc)), 3)


def engine_forward(toolkits, dataloader, evaluator, config):
    (engine, allocate_buffers, inference, postprocess_the_outputs) = toolkits
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine, context)

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
            trt_input = x.numpy()

            inputs[0].host = trt_input.reshape(-1)
            output_shape = (trt_input.shape[0], 1000)

            trt_outputs = inference(context,
                                    bindings=bindings,
                                    inputs=inputs,
                                    outputs=outputs,
                                    stream=stream)

            feat = postprocess_the_outputs(trt_outputs[0], output_shape)

            pred = torch.from_numpy(feat).float()
            torch_sync(config)
            core_time += time.time() - core_time_start

            top1 = evaluator(pred, y)

            all_top1.extend(top1.cpu())

        acc.append(np.mean(all_top1))

    logger.info("Top1 Acc: " + str(acc))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(acc)), 3)
