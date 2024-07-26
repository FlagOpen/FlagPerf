# 参评AI芯片信息

* 厂商：XXX


* 产品名称：XXX
* 产品型号：XXX
* TDP：XXX

# 所用服务器配置

* 服务器数量：XXX
* 单服务器内使用卡数：XXX
* 服务器型号：XXX
* 操作系统版本：XXX
* 操作系统内核：XXX
* CPU：XXX
* docker版本：XXX
* 内存：XXX
* 服务器间AI芯片直连规格及带宽：此评测项不涉及服务期间AI芯片直连

# 算子库版本

https://github.com/FlagOpen/FlagGems. Commit ID: XXX

# 评测结果

## 核心评测结果

| 评测项  | 平均相对误差(with FP64-CPU) | TFLOPS(cpu wall clock) | TFLOPS(kernel clock) | FU(FLOPS Utilization)-cputime | FU-kerneltime |
| ---- | -------------- | -------------- | ------------ | ------ | ----- |
| flaggems | {{ flaggems_average_relative_error }}    | {{ flaggems_tflops }}       | {{ flaggems_kernel_clock}}        | {{ flaggems_fu_cputime }} | {{ flaggems_kerneltime }} |
| nativetorch | {{ nativetorch_average_relative_error }}    | {{ nativetorch_tflops }}      | {{ nativetorch_kernel_clock}}      | {{ nativetorch_fu_cputime }}      | {{ nativetorch_kerneltime }}    |

## 其他评测结果

| 评测项  | 相对误差(with FP64-CPU)标准差 | cputime | kerneltime | cputime吞吐 | kerneltime吞吐 | 无预热时延 | 预热后时延 |
| ---- | -------------- | -------------- | ------------ | ------------ | -------------- | -------------- | ------------ |
| flaggems | {{ flaggems_relative_error }}    | {{ flaggems_cpu_time }}       | {{ flaggems_kernel_time }}        | {{ flaggems_cpu_ops }} | {{ flaggems_kernel_ops }} | {{ flaggems_no_warmup_delay }} | {{ flaggems_warmup_delay }} |
| nativetorch | {{ nativetorch_relative_error }}    | {{ nativetorch_cpu_time }}       | {{ nativetorch_kernel_time }}        | {{ nativetorch_cpu_ops }} | {{ nativetorch_kernel_ops }} | {{ nativetorch_no_warmup_delay }} | {{ nativetorch_warmup_delay }} |

## 能耗监控结果

| 监控项  | 系统平均功耗  | 系统最大功耗  | 系统功耗标准差 | 单机TDP | 单卡平均功耗 | 单卡最大功耗 | 单卡功耗标准差 | 单卡TDP |
| ---- | ------- | ------- | ------- | ----- | ------------ | ------------ | ------------- | ----- |
| nativetorch监控结果 | {{ nativetorch_ave_system_power }} | {{ nativetorch_max_system_power }} | {{ nativetorch_system_power_stddev }}   | /     | {{ nativetorch_single_card_avg_power }}       | {{ nativetorch_single_card_max_power}}      | {{ nativetorch_single_card_power_stddev}}        | {{ nativetorch_single_card_tdp}}  |
| flaggems监控结果 | {{ flaggems_ave_system_power }} | {{ flaggems_max_system_power }} | {{ flaggems_system_power_stddev }}   | /     | {{ flaggems_single_card_avg_power }}       | {{ flaggems_single_card_max_power}}      | {{ flaggems_single_card_power_stddev}}        | {{ flaggems_single_card_tdp}}  |

## 其他重要监控结果

| 监控项  | 系统平均CPU占用 | 系统平均内存占用 | 单卡平均温度 | 单卡最大显存占用 |
| ---- | --------- | -------- | ------------ | -------------- |
| nativetorch监控结果 | {{nativetorch_avg_cpu_usage}}    | {{nativetorch_avg_mem_usage}}   | {{nativetorch_single_card_avg_temp}}       | {{nativetorch_max_gpu_memory_usage_per_card}}        |
| flaggems监控结果 | {{flaggems_avg_cpu_usage}}    | {{flaggems_avg_mem_usage}}   | {{flaggems_single_card_avg_temp}}       | {{flaggems_max_gpu_memory_usage_per_card}}        |
