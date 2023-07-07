from loguru import logger
import torch
import numpy as np
import time


def model_forward(model, dataloader, evaluator, config):
    start = time.time()

    core_time = 0.0

    acc = []

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        all_top1 = []
        for step, (x, y) in enumerate(dataloader):
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))

            with torch.no_grad():
                
                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                

                top1 = evaluator(pred, y)

                all_top1.extend(top1.cpu())
            core_time += time.time() - core_time_start

        acc.append(np.mean(all_top1))

    duration = time.time() - start
    model_forward_perf = config.repeat * len(
        dataloader) * config.batch_size / duration
    logger.info("Model Forward(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " ips")
    model_forward_core_perf = config.repeat * len(
        dataloader) * config.batch_size / core_time
    logger.info("Model Forward(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " ips")

    return acc


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

            top1 = evaluator(pred, y)

            all_top1.extend(top1.cpu())
            core_time += time.time() - core_time_start

        acc.append(np.mean(all_top1))

    duration = time.time() - start
    model_forward_perf = config.repeat * len(
        dataloader) * config.batch_size / duration
    logger.info("Vendor Inference(" + config.vendor + ") Perf: " +
                str(model_forward_perf) + " ips")
    model_forward_core_perf = config.repeat * len(
        dataloader) * config.batch_size / core_time
    logger.info("Vendor Inference(" + config.vendor + ") core Perf: " +
                str(model_forward_core_perf) + " ips")

    return acc
