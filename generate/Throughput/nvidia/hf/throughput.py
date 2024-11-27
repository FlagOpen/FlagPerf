import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from loguru import logger
import yaml
import pandas as pd
import argparse
import gc
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
        "engine"] + "/" + config["chip"] + ".yaml"
    task_config = read_yaml_file(path)

    logpath = config["log_path"]
    logfile = str(logpath + "hf/throughput.log")
    logger.remove()
    logger.add(logfile,
               format="{time} {level} {message}",
               level='INFO',
               rotation='1 day',
               mode='w')

    local_model_path = config["model_path"]
    data_df = pd.read_csv(config["data_path"])
    tasks = data_df["dialogue"].tolist()
    nums = task_config["task_nums"]
    tasks = tasks[0:nums]
    answers = data_df["summary"].tolist()
    answers = answers[0:nums]

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    duration = 0
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                 device_map="auto").eval()

    inputs_all = []
    answers_all = []
    for i in range(len(tasks)):
        if type(tasks[i]) == str:
            inputs_all.append(
                "Please generate a summary about it:" + str(tasks[i]) +
                "Please generate a summary after the article content is finished."
            )
            answers_all.append(answers[i])
    steps = 0
    origin_len = 0
    new_len = 0
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', "rougeLsum"], use_stemmer=True)

    rouge1 = 0
    rouge2 = 0
    rougel = 0
    rougelsum = 0
    lenth = 0
    for firststep in tqdm(range(len(inputs_all))):
        inputs = tokenizer([inputs_all[firststep]],
                           return_tensors="pt",
                           padding=True)
        inputs = inputs.to(device)
        start = time.perf_counter()
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=16,
            num_return_sequences=1,
            early_stopping=False,
            attention_mask=inputs.attention_mask,
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        duration += end - start
        for i in range(len(outputs)):
            origin_len += len(inputs.input_ids[i])
            new_len += 16
            pre_ans = tokenizer.decode(outputs[i][len(outputs[i]) - 16::],
                                       skip_special_tokens=True)
            scores = scorer.score(answers_all[firststep], pre_ans)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougel += scores['rougeL'].fmeasure
            rougelsum += scores['rougeLsum'].fmeasure
    logger.info(f"newly generated tokens:{float(new_len)}")
    logger.info(f"total number of tokens:{float(new_len+origin_len)}")
    logger.info(f"duration:{duration}")
    logger.info(f"token/s :{(origin_len+new_len)/duration}")
    logger.info(f"rougeone:{rouge1/(len(inputs_all))}")
    logger.info(f"rougetwo:{rouge2/(len(inputs_all))}")
    logger.info(f"rougeL:{rougel/(len(inputs_all))}")
    logger.info(f"rougeLsum:{rougelsum/(len(inputs_all))}")

    model = None
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the config file")
    args = parser.parse_args()
    Throughput_Record(args.config)
