from os.path import join as opj
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="flagperf_monitor_vis")
    parser.add_argument("--node_log_dir", type=str, required=True)
    args = parser.parse_args()
    return args


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def smooth(x, factor):
    y = []
    for i in range(factor):
        y.append(x[0])
    for item in x:
        y.append(item)

    z = []
    for i in range(len(x)):
        tmp = 0
        for j in range(factor):
            tmp += y[i + j]
        z.append(tmp / factor)

    return z


args = parse_args()
logdir = args.node_log_dir
results = {}
for index in ['cpu', 'mem', 'pwr']:
    file = open(opj(logdir, index + '.log.txt'))
    result = []
    for line in file.readlines():
        if 'UTC' in line or len(line) < 2 or 'CST' in line:
            continue
        result.append(float(line))
    results[index] = result

    plt.figure(figsize=(8, 6))

    mean = np.mean(results[index])
    upper = np.max(results[index])
    std = np.std(results[index])

    x = []
    for i in range(len(results[index])):
        x.append(i)

    plt.plot(x, results[index], color='orange', alpha=0.5,
             label=index.upper() + '占用' if index != 'pwr' else "整机功耗")
    plt.plot(x, smooth(results[index], 5), color='blue',
             label=(index.upper() + '占用' if index != 'pwr' else "整机功耗") + "平滑处理")
    plt.plot(x, len(results[index]) * [mean], color='red',
             label=(index.upper() + '占用' if index != 'pwr' else "整机功耗") + "均值")
    plt.fill_between(x, [mean + std] * len(results[index]), [mean - std] * len(results[index]), color='salmon',
                     alpha=0.2, label=(index.upper() + '占用' if index != 'pwr' else "整机功耗") + "标准差")

    plt.grid("True")
    plt.legend()
    plt.title(index.upper() + '占用时间曲线图' if index != 'pwr' else "整机功耗时间曲线图")
    plt.xlabel('测试启动时间')
    plt.xticks([])
    plt.ylabel(index.upper() + '占用率' if index != 'pwr' else "整机功耗")

    plt.savefig(index + ".svg", dpi=None)
    plt.show()

npu = {}

file = open(opj(logdir, 'npu.log.txt'))
for i in range(8):
    npu[i] = {'温度': [], '功耗': [], 'HBM占用率': [], '计算占用率': []}
next_npu_id = 0
for line in file.readlines():
    if '910B3' in line:
        info = line.split()
        npu[next_npu_id]['温度'].append(float(info[7]))
        npu[next_npu_id]['功耗'].append(float(info[6]))
    if '65536' in line:
        info = line.split()
        npu[next_npu_id]['HBM占用率'].append(float(info[9]) / float(info[11]))
        npu[next_npu_id]['计算占用率'].append(float(info[5]))
        next_npu_id = (next_npu_id + 1) % 8

for metric in ['温度', '功耗', 'HBM占用率', '计算占用率']:
    for npu_id in range(8):

        mean = np.mean(npu[npu_id][metric])
        upper = np.max(npu[npu_id][metric])
        std = np.std(npu[npu_id][metric])

        x = []
        for i in range(len(npu[npu_id][metric])):
            x.append(i)

        plt.figure(figsize=(8, 6))

        plt.plot(npu[npu_id][metric], color='orange', alpha=0.5, label=metric.upper())
        plt.plot(smooth(npu[npu_id][metric], 10), color='blue', label=metric.upper() + '平滑处理')
        plt.plot([mean] * len(npu[npu_id][metric]), color='red', label=metric.upper() + '均值')
        plt.fill_between(x, [mean + std] * len(npu[npu_id][metric]), [mean - std] * len(npu[npu_id][metric]),
                         color='salmon', alpha=0.2, label=metric.upper() + '标准差')

        plt.grid("True")
        plt.legend()
        plt.title(metric + '时间曲线图')
        plt.xlabel('测试启动时间')
        plt.xticks([])
        plt.ylabel(metric)

        plt.savefig(metric + 'npu' + str(npu_id) + ".svg", dpi=None)
        plt.show()

        print('npu: ', npu_id, metric.upper() + ' Metrics:')
        print('Max ', np.max(npu[npu_id][metric]), 'Mean ', np.mean(npu[npu_id][metric]), 'Var ',
              np.var(npu[npu_id][metric]))

