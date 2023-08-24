import os
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

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
    
    
class mmlu(Dataset):

    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.data_dir, config.weight_dir))
        self.records = []
        self.length = 0
        
        for task in TASKS:
        
            logger.debug("Loading 5-shot " + str(task))
            
            dev_df = pd.read_csv(os.path.join(config.data_dir, config.mmlu_dir, "dev", task + "_dev.csv"), header=None)[:config.few_shots]
            test_df = pd.read_csv(os.path.join(config.data_dir, config.mmlu_dir, "test", task + "_test.csv"), header=None)
            
            for i in range(test_df.shape[0]):
                k = config.few_shots
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task, k)
                prompt = train_prompt + prompt_end
                while len(self.tokenizer.tokenize(prompt)) + 1> 2048:
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = "\n\n".join(prompt_split)
                label = test_df.iloc[i, test_df.shape[1]-1]
                token_prompt = self.tokenizer(prompt, return_tensors="pt")
                token_label = self.tokenizer([label], return_tensors="pt")
                self.records.append({"prompt":token_prompt, "answer":token_label.input_ids})
                self.length += 1
                

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.records[idx]
        
        
def build_dataloader(config):
    dataset = mmlu(config)
    assert config.batch_size == 1
    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=config.num_workers,
                        pin_memory=True)

    return loader