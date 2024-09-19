import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from loguru import logger
import yaml
import pandas as pd
import argparse
import gc
import os


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
    logfile = str(logpath + "hf/ttft.log")
    logger.remove()
    logger.add(logfile,
               format="{time} {level} {message}",
               level='INFO',
               rotation='1 day',
               mode='w')

    local_model_path = config["model_path"]
    data_df = pd.read_csv(config["data_path"])
    tasks = data_df["dialogue"].tolist()
    task_nums = task_config["task_nums"]
    tasks = tasks[0:task_nums]

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    duration = 0
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                 device_map="auto").eval()
    parameter = sum(p.numel() for p in model.parameters())
    logger.info(f"parameter: {float(parameter)}")
    for i in tqdm(range(len(tasks)), desc="inference"):
        with torch.no_grad():
            input_text = tasks[i]
            if type(input_text) != str:
                continue
            inputs = tokenizer([
                "Please generate a summary about it:" + input_text +
                "Please generate a summary after the article content is finished."
            ],
                               return_tensors="pt")
            inputs = inputs.to(device)
            start = time.perf_counter()
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                num_return_sequences=1,
                early_stopping=False,
                attention_mask=inputs.attention_mask,
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            duration += end - start
    logger.info(f"TTFT: {duration}")

    model = None
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the config file")
    args = parser.parse_args()
    TTFT_Record(args.config)
