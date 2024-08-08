# 参评AI芯片信息

* 厂商：Nvidia

* 产品名称：A100
* 产品型号：A100-40GiB-SXM
* TDP：400W

# 所用服务器配置

* 服务器数量：1
* 单服务器内使用卡数: 1
* 服务器型号：DGX A100
* 操作系统版本：Ubuntu 20.04.4 LTS
* 操作系统内核：linux5.4.0-113
* CPU：AMD EPYC7742-64core
* docker版本：20.10.16
* 内存：1TiB
* 服务器间AI芯片直连规格及带宽：此评测项不涉及服务期间AI芯片直连

# 算子库版本

https://github.com/FlagOpen/FlagGems. Commit ID: 7042de1d8fb6f978596322faaeda6b55ca1ae5ec

# 评测结果

## 核心评测结果

| 评测项  | correctness | TFLOPS(cpu wall clock) | TFLOPS(kernel clock) | FU(FLOPS Utilization)-cputime | FU-kerneltime |
| ---- | -------------- | -------------- | ------------ | ------ | ----- |
| flaggems |  True    | 259.29TFLOPS       | 262.34TFLOPS        | 83.11% | 84.08% |
| nativetorch |  True    | 259.62TFLOPS      | 262.02TFLOPS      | 83.21%      | 83.98%    |

## 其他评测结果

| 评测项  | cputime | kerneltime | cputime吞吐 | kerneltime吞吐 | 无预热时延 | 预热后时延 |
| ---- | -------------- | -------------- | ------------ | ------------ | -------------- | -------------- |
| flaggems | 4240.44us       | 4191.23us        | 235.82op/s | 238.59op/s | 24118510.88us | 4329.25us |
| nativetorch | 4235.09us       | 4196.35us        | 236.12op/s | 238.3op/s | 143678.03us | 4232.55us |

## 能耗监控结果

| 监控项  | 系统平均功耗  | 系统最大功耗  | 系统功耗标准差 | 单机TDP | 单卡平均功耗 | 单卡最大功耗 | 单卡功耗标准差 | 单卡TDP |
| ---- | ------- | ------- | ------- | ----- | ------------ | ------------ | ------------- | ----- |
| nativetorch监控结果 | 1430.0W | 1482.0W | 36.77W   | /     | 158.93W       | 189.0W      | 20.27W        | 400W  |
| flaggems监控结果 | 1456.0W | 1482.0W | 36.77W   | /     | 403.5W       | 404.0W      | 0.5W        | 400W  |

## 其他重要监控结果

| 监控项  | 系统平均CPU占用 | 系统平均内存占用 | 单卡平均温度 | 单卡最大显存占用 |
| ---- | --------- | -------- | ------------ | -------------- |
| nativetorch监控结果 | 0.606%    | 1.085%   | 36.21°C       | 2.534%        |
| flaggems监控结果 | 0.648%    | 1.087%   | 45.0°C       | 3.377%        |