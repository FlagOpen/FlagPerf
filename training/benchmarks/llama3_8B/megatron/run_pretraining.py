import subprocess
from argparse import ArgumentParser
import os
import sys
from importlib import import_module


def parse_args():
    '''we parse ddp related args, check system config args, and running env
       args such as --data_dir_xxx. Then pass all useful args to the real
       training script.
    '''
    parser = ArgumentParser(description="flagscale main python")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--node_rank", type=int, required=True)
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=int, required=True)
    parser.add_argument("--vendor", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--flagperf_config_file", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    sys.path.append(os.path.dirname(args.flagperf_config_file))
    config_file = os.path.basename(args.flagperf_config_file).split('.')[0]
    config_dir_path = os.path.dirname(args.flagperf_config_file)

    module = import_module(config_file)

    seqlength = getattr(module, 'seqlength')
    localbs = getattr(module, 'localbs')
    train_steps = getattr(module, 'train_steps')
    theoryflops = getattr(module, 'theoryflops')
    epochs = getattr(module, 'epochs')
    tensor_parallel = getattr(module, 'tensor_parallel')
    pipeline_parallel = getattr(module, 'pipeline_parallel')


    task_log_file = os.path.join(args.log_dir, "megatron.log.txt")

    exec_cmd = "cp arguments.py /workspace/Megatron-LM/megatron/training/;cp tokenizer.py /workspace/Megatron-LM/megatron/training/tokenizer/;bash pretrain_llama3.sh"
    exec_cmd = exec_cmd + " " + os.path.join(config_dir_path, "training_adapter.sh")

    with open(task_log_file, "w") as f:
        p = subprocess.Popen(exec_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        p.wait()

    time_per_step = -1.0
    with open(task_log_file) as f:
        for line in f.readlines():
            if "elapsed time per iteration (ms): " in line:
                info = line.split("|")[2]
                steptime = info.split(":")[1]
                time_per_step = float(steptime) / 1000

    whole_tps = 512* seqlength / time_per_step
    chip_tps = whole_tps / (args.nproc_per_node * args.nnodes)
    print("System tokens per second: ", whole_tps)
    print("Tokens/p/s: ", chip_tps)
    print("MFU: ", chip_tps * 80000000000.0 * 6 / theoryflops)
