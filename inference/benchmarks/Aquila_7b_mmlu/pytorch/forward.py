import os
import time

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

from tools import torch_sync
from flagai.data.tokenizer import Tokenizer
from .utils import TASKS, gen_prompt, format_example, batch_split


def cal_perf(config, tokens, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * tokens / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " tps")
    model_forward_core_perf = config.repeat * tokens / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " tps")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def batch_infer(model, tokenizer, config, prompts):
    answers = []
    start = time.time()
    core_time = 0.0
    prompt_tokens_all = 0
    for batch_input in tqdm(batch_split(prompts, config.batch_size)):
        prompt_tokens = tokenizer.tokenize(batch_input[0])
        tokens = tokenizer.encode(batch_input[0])
        tokens = torch.LongTensor([tokens]).cuda()
        prompt_tokens_all += len(prompt_tokens)
        with torch.no_grad():
            torch_sync(config)
            core_time_start = time.time()
            model_forward_output = model(tokens)
            predict_result = torch.argmax(model_forward_output["logits"], dim=1)
            model_output = tokenizer.decode([predict_result.item()])
            core_time += time.time() - core_time_start
        answers.append(model_output)
    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
    config, prompt_tokens_all, duration, core_time, "Validation")
    answers = [answer[-1] for answer in answers]
    return answers, model_forward_perf , model_forward_core_perf


def model_forward(model, dataloader, evaluator, config):
    run_results = {}
    model_path = os.path.join(os.path.dirname(__file__), config.download_path)
    tokenizer = Tokenizer.from_pretrained(config.model_name, cache_dir=model_path)

    average_model_forward_perf = []
    average_model_forward_core_perf = []

    for task in TASKS:
        logger.info('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev", task + "_dev.csv"), header=None)[:config.ntrain]
        test_df = pd.read_csv(os.path.join(config.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = config.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})

        pred_answers, model_forward_perf, model_forward_core_perf = batch_infer(model, tokenizer, config, [record['prompt'] for record in records])
        
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
        average_model_forward_perf.append(model_forward_perf)
        average_model_forward_core_perf.append(model_forward_core_perf)
    evaluate_result = evaluator(run_results)
    return model_forward_perf, round(np.mean(model_forward_core_perf), 3), evaluate_result


def engine_forward(model, dataloader, evaluator, config):
    if config.compiler is None:
        return None, None, None