# 厂商信息

官网: https://www.metax-tech.com/

沐曦集成电路（上海）有限公司，于2020年9月成立于上海，并在北京、南京、成都、杭州、深圳、武汉和长沙等地建立了全资子公司暨研发中心。沐曦拥有技术完备、设计和产业化经验丰富的团队，核心成员平均拥有近20年高性能GPU产品端到端研发经验，曾主导过十多款世界主流高性能GPU产品研发及量产，包括GPU架构定义、GPU IP设计、GPU SoC设计及GPU系统解决方案的量产交付全流程。

沐曦致力于为异构计算提供全栈GPU芯片及解决方案，可广泛应用于智算、智慧城市、云计算、自动驾驶、数字孪生、元宇宙等前沿领域，为数字经济发展提供强大的算力支撑。

沐曦打造全栈GPU芯片产品，推出曦思®N系列GPU产品用于智算推理，曦云®C系列GPU产品用于通用计算，以及曦彩®G系列GPU产品用于图形渲染，满足“高能效”和“高通用性”的算力需求。沐曦产品均采用完全自主研发的GPU IP，拥有完全自主知识产权的指令集和架构，配以兼容主流GPU生态的完整软件栈（MXMACA®），具备高能效和高通用性的天然优势，能够为客户构建软硬件一体的全面生态解决方案，是“双碳”背景下推动数字经济建设和产业数字化、智能化转型升级的算力基石。



# FlagPerf适配验证环境说明
## 环境配置参考
- 硬件
  - 机器型号: 同泰怡 G658V3
  - 加速卡型号: 曦云®C500 64G  
  - 多机网络类型、带宽: InfiniBand，2x200 Gb/s
- 软件
  - OS版本：Ubuntu 20.04.6
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：2.18.0.8
  - VBIOS：1.0.102.0
  - Docker版本：24.0.7


## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：metax/docker_image/pytorch_2.0/Dockerfile
  - 构建后软件安装脚本：metax/docker_image/pytorch_2.0/pytorch_install.sh

- 核心软件信息 
  - AI框架&相关版本：  
    torch: pytorch-2.0-mc  
    torchvision: torchvision-0.15-mc  
    maca: 2.18.0.8  


## 加速卡监控采集
- 加速卡使用信息采集命令

  ```shell 
  mx_smi
  ```
- 监控项示例：

+---------------------------------------------------------------------------------+  
|&emsp; MX-SMI 2.0.12&emsp; &emsp; &emsp; &emsp; &emsp; Kernel Mode Driver Version: 2.2.0&emsp; &emsp; &emsp; &thinsp; |  
|&emsp;  MACA Version: 2.0&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;  BIOS Version: 1.0.102.0&emsp; &emsp; &emsp; &thinsp; &thinsp; |  
|------------------------------------+---------------------+----------------------+  
|&emsp; GPU&emsp;&emsp;&thinsp; NAME &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| Bus-i&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;| GPU-Util&emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;|  
|&emsp; Temp&emsp;&emsp;Power &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| Memory-Usage&emsp;&thinsp;&thinsp;&thinsp;|&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;&thinsp;|  
|=====================+============+==============|  
| &emsp;0&emsp;&emsp;&thinsp; MXC500   &emsp;&emsp;&thinsp;&emsp;&emsp;&thinsp;&emsp;&emsp;| 0000:1b:00.0 &emsp;&emsp;&thinsp;&thinsp; | 0%&emsp;&emsp;&thinsp;&emsp;&thinsp;&thinsp;&emsp;&thinsp;&thinsp;&emsp;&thinsp;&thinsp;&emsp;&thinsp;&thinsp;|  
| &emsp;35C &emsp;&emsp;&thinsp;56W &emsp;&emsp;&thinsp;&emsp;&emsp;&emsp;&emsp; &thinsp; | 914/65536 MiB &thinsp; &thinsp; &thinsp;    | &emsp;&emsp;&thinsp;  &emsp;&emsp;&thinsp;&thinsp;&emsp;&thinsp;&thinsp;&emsp;&emsp;&emsp;|  
+------------------------------------+---------------------+----------------------+  


- 加速卡使用信息采集项说明

|监控项| 日志文件 | 格式 |
|---|---|---|
|温度| mx_monitor.log | xxx C |
|功耗 |mx_monitor.log | xxx W |
|显存占用大小 |mx_monitor.log |xxx MiB |
|总显存大小 |mx_monitor.log |xxx MiB |
|显存使用率 |mx_monitor.log |xxx % |



