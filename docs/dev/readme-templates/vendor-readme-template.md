# 厂商README文件模版

- 文档位置：每个厂商的REAMDE文档位于training/<vendor>/  目录下
- 文档使用的语言：默认为中文README.md，可提供英文版本README.en.md
- 文档的目的：向用户介绍厂商信息，说明适配FlagPerf测试Case的软、硬件环境信息及加速卡监控采集指标。

## 厂商信息

- 名称
- 厂商的介绍

*//* *厂商的产品、服务、特点、优势等，请写在这里*

*// 可以以列表或表格的形式呈现*

## **FlagPerf适配验证环境说明**

### 环境配置参考

- 硬件
  - 机器型号：例如 NVIDIA_DGX-A100
  - 加速卡型号: 执行 加速卡监控命令后，显示的加速卡型号。 例如 NVIDIA_A100-SXM4-40GB
- 软件
  - OS kernel版本: 
  -   *// 例如* *Linux 5.4.0-136-generic x86_64 【uname -srm 可查看】*

  - Docker 版本
  -   // 例如 20.10.9, 【docker --version可查看】

  - 主机上依赖库的版本和安装方式

- 

### 容器镜像信息

- 容器构建信息
  - Dockerfile路径：<vendor>/docker_image/<framework>/Dockerfile
  - 构建后软件安装脚本：<vendor>/docker_image/<framework>/<framework>_install.sh
  - ​       *// 软件安装、目录创建、文件copy等*
- 核心软件信息（例如cuda等，通常包含在容器的基础镜像中或软件安装脚本<framework>_install.sh）
  - AI框架&版本
  - ​      *//* *例如pip install torch_xmlir --pipsource <pipsource>*

  - 其它软件版本

### 加速卡监控采集

- 加速卡使用信息采集命令

  -  *// 【如有，请填写】**例如xpu-smi, cnmon*

- 加速卡信息采集日志格式说明

  -  *// 根据监控脚本实现情况对日志格式进行说明，方便用户参照*

- 加速卡使用信息采集项说明

  - | 监控项       | 日志文件                | 格式    |
    | ------------ | ----------------------- | ------- |
    | 温度         | accelerator_monitor.log | xxx C   |
    | 功耗         | accelerator_monitor.log | xxx W   |
    | 显存占用大小 | accelerator_monitor.log | xxx MiB |
    | 显存大小     | accelerator_monitor.log | xxx MiB |
    | 加速卡使用率 | accelerator_monitor.log | xxx %   |

  - 

// 监控项示例：

*2023-02-19-16:47:08  # 监控时间*

*48C 311W 28736MiB 40536MiB 83%*

*47C 329W 28880MiB 40536MiB 98%*

*48C 319W 28880MiB 40536MiB 86%*

*48C 271W 28880MiB 40536MiB 80%*

*58C 334W 28880MiB 40536MiB 92%*

*55C 301W 28878MiB 40536MiB 74%*

*57C 210W 28880MiB 40536MiB 97%*

*57C 347W 28736MiB 40536MiB 49%*

*//* *注：以上只是个示例，不要求所有卡都有，请根据厂商自己的实际情况来说明。*