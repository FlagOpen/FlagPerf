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
            p = subprocess.Popen(exec_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
        f.close()

        exec_cmd = getattr(module, "scale_install_cmd")
        logfile = os.path.join(install_logdir, "scale_install.log.txt")
        with open(logfile, 'w') as f:
            p = subprocess.Popen(exec_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
        f.close()
        
        exec_cmd = getattr(module, "energon_locate_cmd")
        logfile = os.path.join(install_logdir, "energon_locate.log.txt")
        with open(logfile, 'w') as f:
            p = subprocess.Popen(exec_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
        f.close()

        with open(logfile, 'r') as f:
            energon_locate = f.readline().replace('\n', '')
        print(energon_locate)

        src_dir = os.path.join(energon_locate, "megatron", "energon")
        dst_dir = os.path.join(getattr(module, "scale_home"), "megatron", "megatron")
        exec_cmd = f"cp -r {src_dir} {dst_dir}/"
        
        logfile = os.path.join(install_logdir, "energon_copy.log.txt")
        with open(logfile, 'w') as f:
            p = subprocess.Popen(exec_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
        f.close()


def replace_yamls(scale_home, config_module, args):
    scale_conf_dir = getattr(config_module, "scale_conf_dir")
    dist_yaml = getattr(config_module, "configyaml")
    with open(dist_yaml, 'r') as f:
        dist_data = yaml.safe_load(f)

    try:
        dist_data["experiment"]["exp_dir"] = os.path.join(args.log_dir, "outputs_llava1.5")
        hosts = args.hosts.split(",")
        dist_data["experiment"]["runner"]["nnodes"] = len(hosts)
        dist_data["experiment"]["runner"]["ssh_port"] = getattr(config_module, "flagscale_ssh_port")
        hostfile = os.path.join(scale_home, "hostfile")
        with open(hostfile, 'w') as f:
            for host in hosts:
                slots = dist_data["experiment"]["runner"]["nproc_per_node"]
                chiptype = getattr(config_module, "flagscale_chip_type")
                f.write(f"{host} slots={slots} type={chiptype}\n")
        dist_data["experiment"]["runner"]["hostfile"] = hostfile
    except Exception as e:
        print(e)
        print("You're using an illegal config.yaml in flagscale. You must fix it")

    print(dist_data)

    train_yaml = getattr(config_module, "trainyaml")

    with open(train_yaml, 'r') as f:
        train_data = yaml.safe_load(f)

    try:
        train_data["system"]["checkpoint"]["save_interval"] = 1000
        train_data["system"]["checkpoint"]["pretrained_checkpoint"] = os.path.join(args.data_dir, "LLaVA_megatron", "vicuna_instruct_clip336_tp1_combined_mcore")

        train_data["model"]["train_iters"] = 5000
        train_data["model"].pop("img_embedding_idx", None)
        train_data["data"]["data_path"] = getattr(config_module, "datasetyaml")
        train_data["data"]["valid_path"] = getattr(config_module, "datasetyaml")
        train_data["data"]["prompt_path"] = getattr(config_module, "prompt")
        train_data["data"]["tokenizer"]["tokenizer_model"] = os.path.join(args.data_dir, "vicuna-7b-v1___5/tokenizer.model")
    except Exception as e:
        print("You're using an illegal trainllava.yaml in flagscale. You must fix it")


    print(train_data)

    dataset_yaml = getattr(config_module, "datasetyaml")
    
    with open(dataset_yaml, 'r') as f:
        dataset_data = yaml.safe_load(f)

    try:
        llava_train_dir = os.path.join(args.data_dir, "LLaVA-Pretrain/wds")
        dataset_data["splits"]["train"]["datasets"][0]["path"] = llava_train_dir
        dataset_data["splits"]["val"]["datasets"][0]["path"] = llava_train_dir
    except Exception as e:
        print("You're using an illegal dataset.yaml in flagscale. You must fix it")
    
    print(dataset_data)

    with open(dist_yaml, 'w') as f:
        yaml.safe_dump(dist_data, f)

    with open(train_yaml, 'w') as f:
        yaml.safe_dump(train_data, f)

    with open(dataset_yaml, 'w') as f:
        yaml.safe_dump(dataset_data, f)


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
        p = subprocess.Popen(exec_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()

    timestamp_log_host = hosts[-1]
    timestamp_log_noderank = len(hosts) - 1

    timestamp_log_file = os.path.join(args.log_dir, "outputs_llava1.5", "logs", "host_" + str(timestamp_log_noderank) + "_" + timestamp_log_host + ".output")

    info_line = []
    while True:
        try:
            with open(timestamp_log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "elapsed time per iteration" in line:
                        info_line.append(line)
        except Exception as e:
            print("Maybe some errors")
        if len(info_line) == 5000:
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
    tps = 2048 * 256 / ave_steptime / args.world_size
    mfu = tps * 7E9 * 6 / getattr(module, "flops")
    print(f"MFU: {mfu}")

