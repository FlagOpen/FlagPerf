import os
from os.path import join as opj

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="flagperf_monitor_vis")
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    return args


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
args = parse_args()
logdir = args.log_dir

path = os.listdir(logdir)
print(path)

for index in ['温度', '功耗', 'HBM使用率', '计算利用率']:
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    mean_array = []
    upper_array = []
    std_array = []

    for node_log in path:
        file = open(opj(logdir, node_log, 'npu.log.txt'))
        npu = {}
        for i in range(8):
            npu[i] = {'温度': [], '功耗': [], 'HBM使用率': [], '计算利用率': []}
        next_npu_id = 0
        for line in file.readlines():
            if '910B3' in line:
                info = line.split()
                npu[next_npu_id]['温度'].append(float(info[7]))
                npu[next_npu_id]['功耗'].append(float(info[6]))
            if '65536' in line:
                info = line.replace('/', ' ').split()
                npu[next_npu_id]['HBM使用率'].append(float(info[8]) / float(info[9]))
                npu[next_npu_id]['计算利用率'].append(float(info[5]))
                next_npu_id = (next_npu_id + 1) % 8
    
        for npu_id in range(8):
            mean = np.mean(npu[npu_id][index])
            upper = np.max(npu[npu_id][index])
            std = np.std(npu[npu_id][index])
            mean_array.append(mean)
            upper_array.append(upper)
            std_array.append(std)
            print(mean, std)
            plt.scatter(mean, std)
    plt.axis([0, np.max(upper_array), 0, 2 * np.max(std_array)])
    plt.xlabel('均值')
    plt.ylabel('标准差')
    plt.title(index + '：最大值为 ' + str(np.max(upper_array)))
    plt.legend()
    plt.savefig(index + ".svg", dpi=None)
    plt.show()

translate = {'cpu': 'CPU占用率', 'mem': '内存占用率', 'pwr': '整机功耗'}

for index in ['cpu', 'mem', 'pwr']:
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    mean_array = []
    upper_array = []
    std_array = []
    for node_log in path:
        results = {}

        file = open(opj(logdir, node_log, index + '.log.txt'))
        result = []
        for line in file.readlines():
            if 'UTC' in line or len(line) < 2 or 'sudo' in line or 'CST' in line:
                continue
            result.append(float(line))
        results[index] = result

        mean = np.mean(results[index])
        upper = np.max(results[index])
        std = np.std(results[index])
        mean_array.append(mean)
        upper_array.append(upper)
        std_array.append(std)

        plt.scatter(mean, std)

    plt.axis([0, np.max(upper_array), 0, 2 * np.max(std_array)])
    plt.xlabel('均值')
    plt.ylabel('标准差')
    plt.title(translate[index] + '：最大值为 ' + str(np.max(upper_array)))
    plt.legend()
    plt.savefig(translate[index] + ".svg", dpi=None)
    plt.show()

