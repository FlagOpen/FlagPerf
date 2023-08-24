from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync


def cal_perf(config, tokens, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * tokens / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " tps")
    model_forward_core_perf = config.repeat * tokens / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " tps")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def model_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    start = time.time()
    core_time = 0.0

    token_cnt = 0
    correct = 0
    whole = 0

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        for step, item in enumerate(dataloader):
            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))
                             
            tokens = item["prompt"].input_ids.cuda()[0]       
                     
            with torch.no_grad():                
                
                torch_sync(config)
                core_time_start = time.time() 
                  
                y = model(tokens)
                
                torch_sync(config)
                core_time += time.time() - core_time_start
                
                token_cnt += len(tokens[0])
                
                pred = y[0]
                r = evaluator(pred, item["answer"])
            
                correct += r
                whole += 1

    logger.info("MMLU" + str(config.few_shots) + "-shots Acc: " + str(correct / whole))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, token_cnt, duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(correct / whole, 3)


def engine_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    start = time.time()
    core_time = 0.0
    foo_time = 0.0

    token_cnt = 0
    correct = 0
    whole = 0

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        for step, item in enumerate(dataloader):
            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))
                             
            tokens = item["prompt"].input_ids[0]     
            model_inputs = [tokens]  
                     
            with torch.no_grad():                
                
                torch_sync(config)
                core_time_start = time.time() 
                  
                y = model(model_inputs)
                
                torch_sync(config)
                core_time += time.time() - core_time_start
                
                foo_time += y[1]
                model_outputs = y[0]
                
                token_cnt += len(tokens[0])
                
                y = model_outputs[0]
                pred = y[0]
                r = evaluator(pred, item["answer"])
            
                correct += r
                whole += 1

    logger.info("MMLU" + str(config.few_shots) + "-shots Acc: " + str(correct / whole))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, token_cnt, duration, core_time - foo_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(correct / whole, 3)
