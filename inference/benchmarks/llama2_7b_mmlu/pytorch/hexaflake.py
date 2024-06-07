import os
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger
from tools import torch_sync
import time

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions'
        ]
choices = ["A", "B", "C", "D"]      
        
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s
    

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt
    


def mmlu(config):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.data_dir, config.weight_dir))
    records = []
    length = 0
    labels = []

    for task in TASKS:
    
        logger.debug("Loading 5-shot " + str(task))
        
        dev_df = pd.read_csv(os.path.join(config.data_dir, config.mmlu_dir, "dev", task + "_dev.csv"), header=None)[:config.few_shots]
        test_df = pd.read_csv(os.path.join(config.data_dir, config.mmlu_dir, "test", task + "_test.csv"), header=None)
        
        for i in range(test_df.shape[0]):
            k = config.few_shots
            label = test_df.iloc[i, test_df.shape[1]-1]
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt = gen_prompt(dev_df, task, k)
            prompt = prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048:
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            records.append(prompt)
            labels.append(label)
    return records, labels

        
def hx_dataloader(config):
    dataset = mmlu(config)
    assert config.batch_size == 1

    return dataset


def hx_model_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    pass


def cal_perf(config, tokens, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * tokens / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " tps")
    model_forward_core_perf = config.repeat * tokens / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " tps")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def hx_engine_forward(model, dataloader, evaluator, config):
    start = time.time()
    core_time = 0.0
    foo_time = 0.0

    token_cnt = 0
    correct = 0
    whole = 0

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))
        data = dataloader[0]
        label = dataloader[1]
        for i in range(len(data)):
            with torch.no_grad():
                torch_sync(config)
                core_time_start = time.time()

                y = model(data[i])

                torch_sync(config)
                core_time += time.time() - core_time_start
                
                token_cnt += y[2]
                foo_time += y[1]
                model_outputs = y[0]

                r = evaluator(model_outputs, label[i])

                correct += r
                whole += 1

    logger.info("MMLU" + str(config.few_shots) + "-shots Acc: " + str(correct / whole))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, token_cnt, duration, core_time - foo_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(correct / whole, 3)


def hx_model(config):
    pass


def hx_export_model(model,config):
    return None


def hx_evaluator(pred, y):
    if pred == y:
        return 1
    else:
        return 0

