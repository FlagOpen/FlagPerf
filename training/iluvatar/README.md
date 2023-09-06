# 厂商信息


上海天数智芯半导体有限公司，简称天数智芯，国内头部通用GPU高端芯片及超级算力系统提供商，致力于开发自主可控、国际领先的高性能通 用 GPU产品，加速AI计算与图形渲染融合，探索通用GPU赶超发展道路，加快建设自主产业生态，打造强大算力引擎。

天数智芯天垓100加速卡是一款基于天垓100芯片的通用GPU加速卡。 天垓100芯片采用通用GPU架构，7纳米制程及2.5D COWOS封装技术，容纳240亿晶体管，制程FP32，FP16，INT32/16/8等多精度数据混合训练，并可提供147 TFLOPS@FP16/BF16的峰值算力。  天垓100加速卡兼容多种主流服务器和主流软件生态，可助力客户实现无痛系统迁移。 

## 产品优势：

- 性能可预期：天垓100的软硬件架构针对通用计算和人工智能而设计，与行业主流GPU产品软硬件架构可类比，采用2.5D COWOS封装技术，丰富的自研指令集全方位支持标量、矢量、张量运算，提供业界领先的高算力和高能效比。

- 开发易迁移 ：天垓100支持国内外标准化的软硬件生态，兼容国内外主流框架及官方算子、常用网络模型和加速库，兼容主流GPU通用计算模型，应用迁移成本低、耗时短、无需重新开发。

- 应用覆盖广： 天垓100聚焦高性能、通用性和灵活性，支持业界前沿算法，目前已有200多个通用计算及人工智能应用落地，数量持续增加，从容面对未来的算法变迁，为人工智能及通用计算和相关垂直应用行业提供匹配行业高速发展的计算力。







# FlagPerf适配验证环境说明
## 环境配置参考
  - 硬件
    - 机器型号： ILUVATAR BI100
    - 加速卡型号: Iluvatar BI-V100 32G
  - 软件
    - OS kernel版本: 
    Linux 5.4.0-148-generic x86_64
    - Docker 版本: 
    20.10.8

## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：iluvatar/docker_image/pytorch/Dockerfile
  - 构建后软件安装脚本：iluvatar/docker_image/pytorch/pytorch_install.sh

- 核心软件信息 
  - AI框架&版本

    torch: 1.13.1+corex.3.1.0

  - 其它软件版本

    cuda: 10.2

    corex: 3.1.0

    torchtext: 0.14.1+corex.3.1.0

    apex: 0.1+corex.3.1.0



## 加速卡监控采集
- 加速卡使用信息采集命令
  
  ```shell 
  ixsmi |grep 'Default'|awk '{print $3,$5,$9,$11,$13}'
  ```
- 监控项示例：
    ```shell
    2023-03-24-10:13:48
    26C 50W 513MiB 32768MiB 0%
    26C 51W 513MiB 32768MiB 0%
    26C 53W 513MiB 32768MiB 0%
    26C 52W 513MiB 32768MiB 0%
    26C 52W 513MiB 32768MiB 0%
    26C 52W 513MiB 32768MiB 0%
    26C 52W 513MiB 32768MiB 0%
    25C 52W 513MiB 32768MiB 0%
    ```

- 加速卡使用信息采集项说明

|监控项| 日志文件 | 格式 |
|---|---|---|
|温度| iluvatar_monitor.log | xxx C |
|功耗 |iluvatar_monitor.log | xxx W |
|显存占用大小 |iluvatar_monitor.log |xxx MiB |
|总显存大小 |iluvatar_monitor.log |xxx MiB |
|显存使用率 |iluvatar_monitor.log |xxx % |



