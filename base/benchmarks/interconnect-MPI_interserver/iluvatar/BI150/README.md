# 参评AI芯片信息

* 厂商：ILUVATAR

## 服务器1

- 产品名称：BI150
- 产品型号：BI150
- TDP：350W

# 所用服务器配置

* 服务器数量：2

## 服务器1

* 单服务器内使用卡数：8
* 服务器型号：R5300 G5
* 操作系统版本：Ubuntu 20.04.6 LTS
* 操作系统内核：linux5.4.0-148-generic
* CPU：Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
* docker版本：20.10.25
* 内存：512GiB
* 服务器间多卡的MPI互联带宽采用多种通信方式组合，无标定互联带宽

# 指标选型

The following are the three performance metrics commonly used
    1. samples/s (algbw): This metric measures the number of samples processed per second, indicating the algorithmic bandwidth. It reflects the computational efficiency of the algorithm.
    2. busbw: This metric represents the bus bandwidth, which measures the data transfer rate across the system's bus. It is crucial for understanding the communication efficiency between different parts of the system.
    3. busbw * 2: This metric is an extension of busbw, accounting for bidirectional data transfer. It doubles the bus bandwidth to reflect the full duplex capability of the system.
The second metric, busbw, is chosen for the following reasons:
    1. This number is obtained applying a formula to the algorithm bandwidth to reflect the speed of the inter-GPU communication. Using this bus bandwidth, we can compare it with the hardware peak bandwidth, independently of the number of ranks used.
    2. We can horizontally compare the MPI of different patterns such as all-gather/all-reduce/reduce-scatter.

## 注意事项

* 运行结果相差较大时，机器没有定频导致的问题，采用定频方式解决，除非重启机器否则只需执行一次：bash /FlagPerf/base/vendors/iluvatar/dvfs.sh 

# 评测结果

## 核心评测结果

| 评测项  | 服务器间多卡的MPI互联带宽测试值(16卡平均) | 服务器间多卡的MPI互联带宽标定值(16卡平均) | 测试标定比例(16卡平均) |
| ---- | -------------- | -------------- | ------------ |
| 评测结果 | 15.56GB/s    | /       | /    |


## 能耗监控结果

| 监控项  | 系统平均功耗  | 系统最大功耗  | 系统功耗标准差 | 单机TDP | 单卡平均功耗(16卡平均) | 单卡最大功耗(16卡最大) | 单卡功耗标准差(16卡最大) | 单卡TDP |
| ---- | ------- | ------- | ------- | ----- | ------------ | ------------ | ------------- | ----- |
| 监控结果 | 1948.2W | 1957.0W | 14.59.0W    | /     | 95.24W       | 97.0W       | 0.40W        | 350W  |

## 其他重要监控结果

| 监控项  | 系统平均CPU占用 | 系统平均内存占用 | 单卡平均温度(16卡平均) | 单卡平均显存占用(16卡平均) |
| ---- | --------- | -------- | ------------ | -------------- |
| 监控结果 | 20.168%    |1.985%   | 34.73°C      | 13.404%        |

使用torch.all_reduce，进行多机多卡的MPI互联操作，计算服务器内MPI互联带宽

* 注：如镜像启动时ssh并未随命令开启，请切换至[容器内启动](https://github.com/FlagOpen/FlagPerf/blob/main/docs/utils/definitions/IN_CONTAINER_LAUNCH.md)