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
    parser.add_argument("--world_size", type=int, required=True)
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

    module = import_module(config_file)

    seqlength = getattr(module, 'seqlength')
    batchsize = getattr(module, 'batchsize')
    accumulate_steps = getattr(module, 'accumulate_steps')
    train_tokens = getattr(module, 'train_tokens')
    theoryflops = getattr(module, 'theoryflops')
    epochs = getattr(module, 'epochs')
    flashattn = getattr(module, 'flashattn')
    tensor_parallel = getattr(module, 'tensor_parallel')
    pipeline_parallel = getattr(module, 'pipeline_parallel')

    train_samples = int((train_tokens * epochs) // seqlength)

    mbs = batchsize
    gbs = batchsize * args.world_size * accumulate_steps // (tensor_parallel *
                                                             pipeline_parallel)

    task_log_file = os.path.join(args.log_dir, "flagscale.log.txt")

    exec_cmd = "bash flagscale_main.sh"
    exec_cmd = exec_cmd + " " + args.data_dir
    exec_cmd = exec_cmd + " " + str(args.world_size)
    exec_cmd = exec_cmd + " " + str(train_samples)
    exec_cmd = exec_cmd + " " + str(tensor_parallel)
    exec_cmd = exec_cmd + " " + str(pipeline_parallel)
    exec_cmd = exec_cmd + " " + str(mbs)
    exec_cmd = exec_cmd + " " + str(gbs)
    exec_cmd = exec_cmd + " " + str(seqlength)
    exec_cmd = exec_cmd + " " + str(flashattn)

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

    whole_tps = gbs * seqlength / time_per_step
    chip_tps = whole_tps / args.world_size
    print("System tokens per second: ", whole_tps)
    print("Tokens/p/s: ", chip_tps)
    print("MFU: ", chip_tps * 7000000000.0 * 6 / theoryflops)
