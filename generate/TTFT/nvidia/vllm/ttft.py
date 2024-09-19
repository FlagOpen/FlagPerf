import torch
from vllm import LLM, SamplingParams
from loguru import logger
import time
import yaml
import gc
import pandas as pd
import argparse
import os
from transformers import AutoModelForCausalLM


def read_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def TTFT_Record(config_path):
    config = read_yaml_file(config_path)
    currentdir = os.path.dirname(os.path.abspath(__file__))
    grandparent_path = os.path.dirname(currentdir)
    grandparent_path = os.path.dirname(grandparent_path)
    grandparent_path = os.path.dirname(grandparent_path)
    path = grandparent_path + "/tasks/" + config["vendor"] + "/" + config[
        "engine"] + "/" + "task.yaml"
    task_config = read_yaml_file(path)

    logpath = config["log_path"]
    logfile = str(logpath + "vllm/ttft.log")
    logger.remove()
    logger.add(logfile,
               format="{time} {level} {message}",
               level='INFO',
               rotation='1 day',
               mode='w')
    local_model_path = config["model_path"]
    print(local_model_path)
    data_df = pd.read_csv(config["data_path"])
    tasks = data_df["dialogue"].tolist()

    tasks = tasks[0:task_config["task_nums"]]

    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    parameter = sum(p.numel() for p in model.parameters())
    logger.info(f"parameter:{float(parameter)}")
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(5)

    model = LLM(model=local_model_path,
                tensor_parallel_size=config["nproc_per_node"],
                trust_remote_code=True,
                tokenizer_mode="auto")
    params2 = SamplingParams(max_tokens=1)
    inputs = [
        "Please generate a summary about it:" + str(i) +
        "Please generate a summary after the article content is finished."
        for i in tasks if type(i) == str
    ]

    start = time.perf_counter()

    outputs = model.generate(list(inputs), params2)

    torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start

    logger.info(f"TTFT:{duration}")
    model = None
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the config file")
    args = parser.parse_args()
    TTFT_Record(args.config)
