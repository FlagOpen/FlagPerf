# 厂商信息

海光 DCU 系列产品以 GPGPU 架构为基础，兼容通用的“类 CUDA” 环境以及国际主流商业计算软件和人工智能软件，软硬件生态丰富，可广泛应用于大数据处理、人工智能、商业计算等应用领域。

海光 DCU 兼容“类 CUDA” 环境， 软硬件生态丰富，典型应用场景下性能指标达到国际上同类型高端产品的水平。

海光 DCU 主要面向大数据处理、商业计算等计算密集型应用领域，以及人工智能、 泛人工智能类运算加速领域。

# FlagPerf适配验证环境说明
## 环境配置参考
  - 硬件
    - 机器型号：K100 标准机
    - 加速卡型号: K100 64G
  - 软件
    - OS版本：centos 7.6
    - OS kernel版本: 4.18.0-348.el8.0.2.x86_64
    - Docker 版本: 24.0.7

## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：training/dcu/docker_image/\<framework\>/Dockerfile
  - 构建后软件安装脚本: training/dcu/docker_image/\<framework\>/\<framework\>_install.sh

- 核心软件信息

  - AI框架&版本
    - torch: 1.13.1

  - 其它软件版本
    - dtk: 23.10.1


## 加速卡监控采集
- 加速卡使用信息采集命令

  dcu_monitor.py中79行需要修改为实际source的地址

  ```
  source /path/of/dtk/env.sh
  rocm-smi
  ```

- 监控项示例：

    ```
    ============================ System Management Interface =============================
    ======================================================================================
    DCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      DCU%      Mode     
    0       53.0C    96.0W      auto     300.0W     0%         0%        Normal   
    1       53.0C    96.0W      auto     300.0W     0%         0%        Normal   
    2       54.0C    95.0W      auto     300.0W     0%         0%        Normal   
    3       55.0C    96.0W      auto     300.0W     0%         0%        Normal   
    4       54.0C    97.0W      auto     300.0W     0%         0%        Normal   
    5       54.0C    95.0W      auto     300.0W     0%         0%        Normal   
    6       55.0C    93.0W      auto     300.0W     0%         0%        Normal   
    7       54.0C    96.0W      auto     300.0W     0%         0%        Normal   
    ======================================================================================
    =================================== End of SMI Log ===================================
    ```

- 加速卡使用信息采集项说明

|监控项| 日志文件 |
|---|---|
|VRAM(%) | dcu_monitor.log |
|DCU(%) | dcu_monitor.log |