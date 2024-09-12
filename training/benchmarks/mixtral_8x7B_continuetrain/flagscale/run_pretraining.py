import subprocess
from argparse import ArgumentParser
import os
import sys
from importlib import import_module
import yaml
import time


def parse_args():
    '''we parse ddp related args, check system config args, and running env
       args such as --data_dir_xxx. Then pass all useful args to the real
       training script.
    '''
    parser = ArgumentParser(description="flagscale main python")
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--vendor", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--hosts", type=str, required=True)
    parser.add_argument("--host_addr", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--flagperf_config_file", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def install_scale(module, log_dir, debug_mode=False):
    if not debug_mode:
        exec_cmd = getattr(module, "scale_download_cmd")
        print(exec_cmd)

        install_logdir = os.path.join(log_dir, "install_logs")
        os.makedirs(install_logdir)

        logfile = os.path.join(install_logdir, "scale_download.log.txt")
        with open(logfile, 'w') as f:
            p = subprocess.Popen(exec_cmd,
                                 shell=True,
                                 stdout=f,
                                 stderr=subprocess.STDOUT)
        p.wait()
        f.close()

        exec_cmd = getattr(module, "scale_install_cmd")
        logfile = os.path.join(install_logdir, "scale_install.log.txt")
        with open(logfile, 'w') as f:
            p = subprocess.Popen(exec_cmd,
                                 shell=True,
                                 stdout=f,
                                 stderr=subprocess.STDOUT)
        p.wait()
        f.close()


def replace_yamls(scale_home, config_module, args):
    scale_conf_dir = getattr(config_module, "scale_conf_dir")
    dist_yaml = getattr(config_module, "configyaml")
    with open(dist_yaml, 'r') as f:
        dist_data = yaml.safe_load(f)

    try:
        dist_data["experiment"]["exp_dir"] = os.path.join(
            args.log_dir, "outputs_mixtral")
        hosts = args.hosts.split(",")
        dist_data["experiment"]["runner"]["nnodes"] = len(hosts)
        dist_data["experiment"]["runner"]["nproc_per_node"] = args.world_size // len(hosts)
        dist_data["experiment"]["runner"]["ssh_port"] = getattr(
            config_module, "flagscale_ssh_port")
        hostfile = os.path.join(scale_home, "hostfile")
        with open(hostfile, 'w') as f:
            for host in hosts:
                slots = dist_data["experiment"]["runner"]["nproc_per_node"]
                chiptype = getattr(config_module, "flagscale_chip_type")
                f.write(f"{host} slots={slots} type={chiptype}\n")
        dist_data["experiment"]["runner"]["hostfile"] = hostfile
        dist_data["experiment"]["cmds"] = getattr(config_module, "cmds")
    except Exception as e:
        print(e)
        print(
            "You're using an illegal config.yaml in flagscale. You must fix it"
        )

    print(dist_data)

    train_yaml = getattr(config_module, "trainyaml")

    with open(train_yaml, 'r') as f:
        train_data = yaml.safe_load(f)

    try:
        train_data["system"]["checkpoint"]["load"] = os.path.join(args.data_dir, getattr(config_module, "ckpt"))
        train_data["system"]["checkpoint"]["finetune"] = True
        train_data["model"]["train_iters"] = getattr(config_module, "steps")
        train_data["model"]["global_batch_size"] = 64
        train_data["model"]["optimizer"]["lr"] = 1e-6
        train_data["model"]["optimizer"]["lr_scheduler"]["min_lr"] = 1e-7
        train_data["data"]["data_path"] = os.path.join(
            args.data_dir, getattr(config_module, "dataset"),
            "dedup-md5-pile-pile-cc_text_document")
        train_data["data"]["tokenizer"]["tokenizer_path"] = os.path.join(
            args.data_dir, getattr(config_module, "tokenizer"))
    except Exception as e:
        print(e)
        print(train_data)
        print(
            "You're using an illegal trainmixtral.yaml in flagscale. You must fix it"
        )

    print(train_data)

    with open(dist_yaml, 'w') as f:
        yaml.safe_dump(dist_data, f)

    with open(train_yaml, 'w') as f:
        yaml.safe_dump(train_data, f)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    host = args.host_addr
    hosts = args.hosts.split(",")
    print(host, hosts)

    if host != hosts[0]:
        exit(0)

    sys.path.append(os.path.dirname(args.flagperf_config_file))
    config_file = os.path.basename(args.flagperf_config_file).split('.')[0]

    module = import_module(config_file)
    print(module)
    scale_home = getattr(module, "scale_home")

    install_scale(module, args.log_dir)

    replace_yamls(scale_home, module, args)

    scale_conf_dir = getattr(module, "scale_conf_dir")
    configyaml = getattr(module, "configyaml")
    configname = os.path.splitext(os.path.basename(configyaml))[0]
    exec_cmd = f"cd {scale_home}; python3 run.py --config-path {scale_conf_dir} --config-name {configname}"

    print(exec_cmd)
    with open(os.path.join(args.log_dir, "flagscale_main.log.txt"), 'w') as f:
        p = subprocess.Popen(exec_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        p.wait()

    timestamp_log_host = hosts[-1]
    timestamp_log_noderank = len(hosts) - 1

    timestamp_log_file = os.path.join(
        args.log_dir, "outputs_mixtral", "logs", "host_" +
        str(timestamp_log_noderank) + "_" + timestamp_log_host + ".output")

    while True:
        info_line = []
        try:
            with open(timestamp_log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "elapsed time per iteration" in line:
                        info_line.append(line)
        except Exception as e:
            print("Maybe some errors")
        if len(info_line) == getattr(module, "steps"):
            break
        time.sleep(300)

    infos = []
    for line in info_line:
        info = line.split("|")[2]
        steptime = info.split(":")[1]
        stepsecond = float(steptime) / 1000
        infos.append(stepsecond)
    print(infos)

    ave_steptime = sum(infos[1:]) / len(infos[1:])
    tps = 4096 * 64 / ave_steptime / args.world_size
    mfu = tps * 12E9 * 6 / getattr(module, "flops")
    print(ave_steptime, tps)
    print(f"MFU: {mfu}")
