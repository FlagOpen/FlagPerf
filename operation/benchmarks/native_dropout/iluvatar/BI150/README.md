# 参评AI芯片信息

* 厂商：ILUVATAR

* 产品名称：BI150
* 产品型号：BI150
* TDP：W

# 所用服务器配置

* 服务器数量：1


* 单服务器内使用卡数：1
* 服务器型号：
* 操作系统版本：Ubuntu 20.04.6 LTS
* 操作系统内核：linux5.4.0-148-generic
* CPU：
* docker版本：20.10.25
* 内存：
* 服务器间AI芯片直连规格及带宽：此评测项不涉及服务期间AI芯片直连

# 算子库版本
FlagGems:>联系邮箱: contact-us@iluvatar.com获取版本(FlagGems最新适配版本)

# 评测结果

## 核心评测结果

| 评测项  | correctness | TFLOPS(cpu wall clock) | TFLOPS(kernel clock) | FU(FLOPS Utilization)-cputime | FU-kerneltime |
| ---- | -------------- | -------------- | ------------ | ------ | ----- |
| flaggems | True    | 0.06TFLOPS       | 0.06TFLOPS        | 0.12% | 0.12% |
| nativetorch | True    | 0.06TFLOPS      | 0.06TFLOPS      | 0.12%      | 0.12%    |

## 其他评测结果

| 评测项  | cputime | kerneltime | cputime吞吐 | kerneltime吞吐 | 无预热时延 | 预热后时延 |
| ---- | -------------- | -------------- | ------------ | ------------ | -------------- | -------------- |
| flaggems | 27739.55us       | 27755.29us        | 36.05op/s | 36.03op/s | 610123.46us | 10154.52us |
| nativetorch | 29083.64us       | 29060.53us        | 34.38op/s | 34.41op/s | 24458.47us | 11345.64us |

## 能耗监控结果

| 监控项  | 系统平均功耗  | 系统最大功耗  | 系统功耗标准差 | 单机TDP | 单卡平均功耗 | 单卡最大功耗 | 单卡功耗标准差 | 单卡TDP |
| ---- | ------- | ------- | ------- | ----- | ------------ | ------------ | ------------- | ----- |
| nativetorch监控结果 | 2161.62W | 2166.0W | 15.19W   | /     | 140.91W       | 141.0W      | 1.52W        | 350W  |
| flaggems监控结果 | 2180.25W | 2185.0W | 15.75W   | /     | 154.07W       | 155.0W      | 1.79W        | 350W  |

## 其他重要监控结果

| 监控项  | 系统平均CPU占用 | 系统平均内存占用 | 单卡平均温度 | 单卡最大显存占用 |
| ---- | --------- | -------- | ------------ | -------------- |
| nativetorch监控结果 | 99.907%    | 2.173%   | 43.37°C       | 28.864%        |
| flaggems监控结果 | 99.788%    | 2.172%   | 46.25°C       | 25.739%        |