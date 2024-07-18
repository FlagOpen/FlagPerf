# cambricon mlu import
try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    pass
import os
import sys
import torch
import random
import json
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model.builder import load_pretrained_model
from utils.mm_utils import get_model_name_from_path

from evaluate.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT
from evaluate.model_utils import call_llava_engine_df, llava_image_processor
from evaluate.eval_utils import evaluate, parse_multi_choice_response, parse_open_response, calculate_ins_level_acc


def run_model(samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_eval(output_path, answer_path):

    output_dict = json.load(open(output_path))
    answer_dict = json.load(open(answer_path))

    # group by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({data_id: parsed_pred})

    # group by category
    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({data_id: parsed_pred})

    evaluation_result = {}

    for category in CAT_SHORT2LONG.values():
        # get cat_outputs and cat_answers
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            print("Skipping {} for not found".format(category))
            continue
        
        exampels_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            if data_id in cat_answers:
                question_type = cat_answers[data_id]['question_type']
                if question_type != 'multiple-choice':
                    parsed_pred = parse_open_response(parsed_pred) # mainly for type consistency (make it number, etc.)
                else:
                    parsed_pred = parsed_pred

                exampels_to_eval.append({
                    "id": data_id,
                    "question_type": question_type,
                    "answer": cat_answers[data_id]['ground_truth'],
                    "parsed_pred": parsed_pred
                })

        judge_dict, metric_dict = evaluate(exampels_to_eval)
        metric_dict.update({"num_example": len(exampels_to_eval)})

        evaluation_result[category] = metric_dict

    printable_results = {}
    # pdb.set_trace()
    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats: # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                  "acc": round(in_domain_ins_acc, 3)
                                                  }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                           "acc": round(cat_results['acc'], 3)
                                           }
        
    # table.append(["-----------------------------", "-----", "----"])
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {"num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
                                    "acc": round(all_ins_acc, 3)
                                    }

    # print(printable_results['Overall']['acc'])
    return printable_results['Overall']['acc']

def eval_mmmu_llava(model_path, data_path, config_path, output_path, answer_path):
    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(42)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    config = load_yaml(config_path)
    for key, value in config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(data_path, subject, split='validation')
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load model
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(model_path, None,
                                                                model_name)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(samples, model, call_model_engine, tokenizer, processor)
    save_json(output_path, out_samples)
    mmmu = main_eval(output_path, answer_path)
    print("MMMU :", mmmu)

if __name__ == "__main__":
    _, model_path, data_path, config_path, output_path, answer_path = sys.argv
    eval_mmmu_llava(model_path, data_path, config_path, output_path, answer_path)
