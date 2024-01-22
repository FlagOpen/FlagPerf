# 厂商信息
昆仑芯（北京）科技有限公司前身为百度智能芯片及架构部，于2021年4月完成独立融资，首轮估值约130亿元。公司团队在国内最早布局AI加速领域，深耕十余年，是一家在体系结构、芯片实现、软件系统和场景应用均有深厚积累的AI芯片企业。

R480-X8是昆仑芯推出的一款基于2代芯片的的AI加速器组，其采用一块通用基板R480，搭载8颗昆仑芯AI加速模块R300。R480是一款符合OAI（Open Accelerator Infrastructure）系列技术标准的通用基板，R300可基于R480通用基板实现芯片间互联，并且在单节点内8个芯片可组成2个通信环路，提供200GB/s片间互联双向聚合带宽。
R480-X8基于多芯片间高速互联技术，单机可提供高达1 Peta Ops @FP16的AI算力和256G显存，聚合通信带宽高达200GBps，可构建大规模并行计算集群，支持大型模型训练和推理需求。

## 产品优势：

- 灵活易用：昆仑芯SDK可提供从底层驱动环境到上层模型转换等全栈的软件工具。

- 生态完备 ：已与多款通用处理器、操作系统、AI框架完成端到端的适配。

- 规模部署： 已在互联网、智慧工业、智慧交通、智慧金融等场景均有规模落地案例，实际部署数万片。




# FlagPerf适配验证环境说明
## 环境配置参考
- 硬件
  - 机器型号: 昆仑芯AI加速器组R480-X8
  - 加速卡型号: 昆仑芯AI加速卡R300
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- 软件
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：4.0.25
  - Docker镜像和版本：pytorch1.12.1-cpu-ubuntu20.04:v0.01
  - 训练框架版本: xmlir 【[xmlir下载](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xacc-0.1.0-cp38-cp38-linux_x86_64.whl)】
  - 训练编译器版本: xacc 【[xacc下载](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl)】
  - 依赖软件版本：pytorch-1.12.1+cpu

## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：kunlunxin/docker_image/pytorch/Dockerfile
  - 构建后软件安装脚本：kunlunxin/docker_image/pytorch/pytorch_install.sh

- 核心软件信息 
  - AI框架&版本

    torch: pytorch-1.12.1+cpu


## 加速卡监控采集
- 加速卡使用信息采集命令

  ```shell 
  xpu_smi | grep xpu | awk '{print $29"C",$27"W",$22"MiB",$24"MiB",$14"%"}'
  ```
- 监控项示例：
    ```shell
    37C 74W 0MiB 32768MiB 0%
    40C 114W 28653MiB 32768MiB 94%
    34C 74W 0MiB 32768MiB 0%
    37C 71W 0MiB 32768MiB 0%
    43C 99W 28934MiB 32768MiB 99%
    39C 74W 0MiB 32768MiB 0%
    46C 114W 23730MiB 32768MiB 100%
    38C 73W 0MiB 32768MiB 0%
    ```

- 加速卡使用信息采集项说明

|监控项| 日志文件 | 格式 |
|---|---|---|
|温度| kunlunxin_monitor.log | xxx C |
|功耗 |kunlunxin_monitor.log | xxx W |
|显存占用大小 |kunlunxin_monitor.log |xxx MiB |
|总显存大小 |kunlunxin_monitor.log |xxx MiB |
|显存使用率 |kunlunxin_monitor.log |xxx % |



