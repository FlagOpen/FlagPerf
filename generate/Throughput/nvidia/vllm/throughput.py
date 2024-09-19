import torch
from vllm import LLM, SamplingParams
from loguru import logger
import time
import yaml
import gc
import pandas as pd
import argparse
from rouge_score import rouge_scorer
import os


def read_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def Throughput_Record(config_path):
    config = read_yaml_file(config_path)
    currentdir = os.path.dirname(os.path.abspath(__file__))
    grandparent_path = os.path.dirname(currentdir)
    grandparent_path = os.path.dirname(grandparent_path)
    grandparent_path = os.path.dirname(grandparent_path)
    path = grandparent_path + "/tasks/" + config["vendor"] + "/" + config[
        "engine"] + "/" + "task.yaml"
    task_config = read_yaml_file(path)

    logpath = config["log_path"]
    logfile = str(logpath + "vllm/throughput.log")
    logger.remove()
    logger.add(logfile,
               format="{time} {level} {message}",
               level='INFO',
               rotation='1 day',
               mode='w')
    local_model_path = config["model_path"]

    data_df = pd.read_csv(config["data_path"])
    tasks = data_df["dialogue"].tolist()
    answers = data_df["summary"].tolist()
    nums = task_config["task_nums"]
    tasks = tasks[0:nums]

    model = LLM(model=local_model_path,
                tensor_parallel_size=config["nproc_per_node"],
                trust_remote_code=True,
                tokenizer_mode="auto")
    params2 = SamplingParams(max_tokens=16)
    inputs = [
        "Please generate a summary about it:" + str(i) +
        "Please generate a summary after the article content is finished."
        for i in tasks if type(i) == str
    ]
    answers_all = [
        answers[i] for i in range(len(tasks)) if type(tasks[i]) == str
    ]

    start = time.perf_counter()

    outputs = model.generate(list(inputs), params2)
    torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start

    new_generate = 0
    origin_gen = 0
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', "rougeLsum"], use_stemmer=True)
    gentext = []
    for output in outputs:
        generated_text = output.outputs[0].text
        new_generate += float(len(output.outputs[0].token_ids))
        origin_gen += float(len(output.prompt_token_ids))
        gentext.append(generated_text)

    genersum = origin_gen + new_generate
    rouge1 = 0
    rouge2 = 0
    rougel = 0
    rougelsum = 0
    for i in range(len(gentext)):
        scores = scorer.score(answers_all[i], gentext[i])
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure
        rougelsum += scores['rougeLsum'].fmeasure
    logger.info(f"newly generated tokens:{float(new_generate)}")
    logger.info(f"total number of tokens:{float(genersum)}")
    logger.info(f"duration:{duration}")
    logger.info(f"token/s :{genersum/duration}")
    logger.info(f"rougeone:{rouge1/len(gentext)}")
    logger.info(f"rougetwo:{rouge2/len(gentext)}")
    logger.info(f"rougeL:{rougel/len(gentext)}")
    logger.info(f"rougeLsum:{rougelsum/len(gentext)}")

    model = None
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the config file")
    args = parser.parse_args()
    Throughput_Record(args.config)
