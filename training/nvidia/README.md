# 厂商信息

官网: https://www.nvidia.cn

NVIDIA 是人工智能计算领域的领导者。率先采用加速计算，以应对重大挑战。英伟达为当代艺术家和科学家开发先进技术。在 AI 领域的研究成果正在推动众多行业实现转型，并为社会带来深远影响。这些行业包括游戏、医疗健康和交通运输等，总价值达 100 万亿美元。
在全栈和数据中心级别实现加速计算，英伟达打造的类似于一个计算堆栈或神经网络，其中包含硬件、系统软件、平台软件和应用四层。每一层都对计算机制造商、服务提供商和开发者开放，让他们以更适合的方式集成到其产品当中。
[来源](https://images.nvidia.cn/nvimages/aem-dam/zh_cn/Solutions/about-us/documents/NVIDIA-Story-zhCN.pdf)

# FlagPerf适配验证环境说明
## 环境配置参考
  - 硬件
    - 机器型号： NVIDIA DGX A100(40G)
    - 加速卡型号: NVIDIA_A100-SXM4-40GB
    - CPU型号：AMD EPYC7742-64core@1.5G
    - 多机网络类型、带宽: InfiniBand，2x200 Gb/s
  - 软件
    - OS版本：Ubuntu 20.04
    - OS kernel版本: Linux 5.4.0-113-generic
    - 加速卡驱动版本：470.129.06
    - Docker 版本: 20.10.16

## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：training/nvidia/docker_image/\<framework\>/Dockerfile
  - 构建后软件安装脚本: training/nvidia/docker_image/\<framework\>/\<framework\>_install.sh

- 核心软件信息

  - AI框架&版本
    - PyTorch: 1.8.0a0+52ea372
    - paddle: 2.4.0-rc0
    - TesorFlow2: 2.6.0

  - 其它软件版本
    - cuda: 11.4


## 加速卡监控采集
- 加速卡使用信息采集命令

  ```bash
  nvidia-smi | grep 'Default'| awk '{print $3,$5,$9,$11,$13}'
  ```
- 监控项示例：
    ```bash
    2023-03-27-11:20:05
    31C 57W 15MiB 40536MiB 0%
    30C 54W 74MiB 40536MiB 3%
    30C 59W 506MiB 40536MiB 4%
    31C 61W 526MiB 40536MiB 10%
    36C 55W 0MiB 40536MiB 0%
    34C 59W 0MiB 40536MiB 0%
    33C 54W 0MiB 40536MiB 0%
    35C 56W 0MiB 40536MiB 0%
    ```
- 加速卡使用信息采集项说明

|监控项| 日志文件 | 格式 |
|---|---|---|
|温度| nvidia_monitor.log | xxx C |
|功耗 |nvidia_monitor.log | xxx W |
|显存占用大小 |nvidia_monitor.log |xxx MiB |
|总显存大小 |nvidia_monitor.log |xxx MiB |
|显存使用率 |nvidia_monitor.log |xxx % |
