#!/usr/bin/env python3

# pms.py: flagPerf Monitor Statistics
from argparse import ArgumentParser

def get_max_mem(file_path):
    mem_list = []
    with open(file_path) as f:
        lines = f.readlines()

        for line in lines:
            if "MiB" not in line:
                continue
            items = line.split()
            mem_list.append(float(items[2].rstrip("MiB")))

    return round(max(mem_list) / 1024, 1)


def get_max_pow(file_path):
    pwr_list = []
    with open(file_path) as f:
        lines = f.readlines()

        for line in lines:
            if "MiB" not in line:
                continue
            items = line.split()
            if items[-1] == "0%":
                continue
            pwr_list.append(round(float(items[1].rstrip("W"))))

    return str(max(pwr_list))


def get_avg_pow(file_path):
    pwr_list = []
    with open(file_path) as f:
        lines = f.readlines()

        for line in lines:
            if "MiB" not in line:
                continue
            items = line.split()
            if items[-1] == "0%":
                continue
            pwr_list.append(float(items[1].rstrip("W")))

    return str(round(sum(pwr_list) / len(pwr_list)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="nvidia_monitor.log", help="log file path")
    parser.add_argument("-m", "--metric", type=str, help="mem/p")
    args = parser.parse_args()
    if args.metric == "p":
        res_max = get_max_pow(args.file)
        res_avg = get_avg_pow(args.file)
        print(f"avg power : {res_avg}W")
        print(f"max power : {res_max}W")
    elif args.metric == "mem":
        res = get_max_mem(args.file)
        print(f"max mem : {res}G")
    else:
        res_max = get_max_pow(args.file)
        res_avg = get_avg_pow(args.file)
        print(f"avg power : {res_avg}W")
        print(f"max power : {res_max}W")
        res = get_max_mem(args.file)
        print(f"max mem : {res}G")
