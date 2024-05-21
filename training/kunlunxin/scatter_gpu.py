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
plt.rcParams['axes.unicode_minus'] = False
args = parse_args()
logdir = args.log_dir

path = os.listdir(logdir)
print(path)

for index in ['温度', '功耗', '显存使用率', 'GPU利用率']:
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    mean_array = []
    upper_array = []
    std_array = []

    for node_log in path:
        file = open(opj(logdir, node_log, 'xpu.log.txt'))
        gpu = {}
        for i in range(8):
            gpu[i] = {'温度': [], '功耗': [], '显存使用率': [], 'GPU利用率': []}
        next_gpu_id = 0
        for line in file.readlines():
            if '32768' in line:
                info = line.split(' ')
                gpu[next_gpu_id]['温度'].append(float(info[0]))
                gpu[next_gpu_id]['功耗'].append(float(info[1]))
                gpu[next_gpu_id]['显存使用率'].append(float(info[2]) / float(info[3]))
                gpu[next_gpu_id]['GPU利用率'].append(float(info[4]) / 100)

            next_gpu_id = (next_gpu_id + 1) % 8
        for gpu_id in range(8):
            mean = np.mean(gpu[gpu_id][index])
            upper = np.max(gpu[gpu_id][index])
            std = np.std(gpu[gpu_id][index])
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
            if 'UTC' in line or 'CST' in line or 'MB' in line or len(line) < 2:
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

