from importlib import import_module
import subprocess
import time
import tokenizers
import argparse
from model import *
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


import os
import sys

def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Reserved for deepspeed framework")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--flagperf_config", type=str)
    parser.add_argument("--node_rank",
                        type=int,
                        required=True,
                        help="The rank of the node for multi-node distributed training.")
    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="how many hosts to run the testcase.")
    parser.add_argument("--nproc_per_node",
                        type=int,
                        required=True,
                        help="how many processes will run on each host.")
    parser.add_argument("--deepspeed",type=str)
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args, remaining_argv = arg_parser.parse_known_args()
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    sys.path.append(os.path.dirname(args.flagperf_config))
    config_file = os.path.basename(args.flagperf_config).split('.')[0]
    module = import_module(config_file)
    theoryflops = getattr(module, 'theoryflops')

    # stage-1 pretrain
    pretrain_ds_config = os.path.join(script_dir, "config/ds_config_pretrain.json")
    per_device_train_batch_size = getattr(
        module, 'pretrain_per_device_train_batch_size')
    gradient_accumulation_steps = getattr(
        module, 'pretrain_gradient_accumulation_steps')
    
    pretrain_script_args = [str(pretrain_ds_config), str(args.data_dir), str(per_device_train_batch_size), str(gradient_accumulation_steps)]
    
    with open("tokens.txt", "w") as f:
        pass

    start_time_pretrain = time.time()
    subprocess.run(["/bin/bash", "scripts/pretrain.sh"] + pretrain_script_args, check=True)
    end_time_pretrain = time.time()
    pretrain_time = end_time_pretrain - start_time_pretrain
    print("pretrain time is", pretrain_time)

    with open('tokens.txt', 'r') as file:
        lines = file.readlines()
    numbers = [int(line.strip()) for line in lines if line.strip()]
    if numbers:
        tokens_pretrain = sum(numbers) / len(numbers)

    print("tokens_pretrain is", tokens_pretrain)

    with open("tokens.txt", "w") as f:
        pass

    # stage-2 finetune
    finetune_ds_config = os.path.join(script_dir, "config/ds_config_finetune.json")
    per_device_train_batch_size = getattr(
        module, 'finetune_per_device_train_batch_size')
    gradient_accumulation_steps = getattr(
        module, 'finetune_gradient_accumulation_steps')
    
    finetune_script_args = [str(finetune_ds_config), str(args.data_dir), str(per_device_train_batch_size), str(gradient_accumulation_steps)]
    start_time_finetune = time.time()
    subprocess.run(["/bin/bash", "scripts/finetune.sh"] + finetune_script_args, check=True)
    end_time_finetune = time.time()
    finetune_time = end_time_finetune - start_time_finetune
    print("finetune time is", finetune_time)

    with open('tokens.txt', 'r') as file:
        lines = file.readlines()
    numbers = [int(line.strip()) for line in lines if line.strip()]
    if numbers:
        tokens_finetune = sum(numbers) / len(numbers)
        
    print("tokens_finetune is", tokens_finetune)

    # evaluate
    mmmu_model_path = os.path.join(args.data_dir,
                                    getattr(module, 'output_dir_finetune'))
    mmmu_data_path = os.path.join(args.data_dir,
                                    getattr(module, 'mmmu_data_path'))
    mmmu_config_path = os.path.join(script_dir, "config/llava1.5.yaml")
    mmmu_output_path = os.path.join(script_dir, "config/llava1.5_13b.json")
    mmmu_answer_path = os.path.join(script_dir,
                                    "config/answer_dict_val.json")
    subprocess.run([
        "python3", "evaluate/evaluator.py", mmmu_model_path,
        mmmu_data_path, mmmu_config_path, mmmu_output_path,
        mmmu_answer_path
    ])
    whole_tps_pretrain = (tokens_pretrain * 558128) / pretrain_time  # 714
    chip_tps_pretrain = whole_tps_pretrain / (args.nproc_per_node * args.nnodes)
    print("Pretrain stage")
    print("System tokens per second: ", whole_tps_pretrain)
    print("Tokens/p/s: ", chip_tps_pretrain)
    TFLOPS = int(theoryflops / 1000000000000)
    print("Theory TFLOPS: ", TFLOPS)
    print("Tokens/TFLOPS: ", chip_tps_pretrain / TFLOPS)
    print("MFU: ", chip_tps_pretrain * 13000000000.0 * 2 / theoryflops)
    whole_tps_finetune = (tokens_finetune * 665344) / finetune_time
    chip_tps_finetune = whole_tps_finetune / (args.nproc_per_node * args.nnodes)
    print("Finetune stage")
    print("System tokens per second: ", whole_tps_finetune)
    print("Tokens/p/s: ", chip_tps_finetune)
    TFLOPS = int(theoryflops / 1000000000000)
    print("Theory TFLOPS: ", TFLOPS)
    print("Tokens/TFLOPS: ", chip_tps_finetune / TFLOPS)
    print("MFU: ", chip_tps_finetune * 13000000000.0 * 6 / theoryflops)

    total_time = pretrain_time + finetune_time
    mfu_average = (tokens_pretrain * 558128 * 13000000000.0 * 2 +
                    tokens_finetune * 665344 * 13000000000.0 *
                    6) / total_time / (args.nproc_per_node *
                                        args.nnodes) / theoryflops
    print("two-stage average")
    print("MFU: ", mfu_average)
    print("Actual computing power: ", mfu_average * TFLOPS)

