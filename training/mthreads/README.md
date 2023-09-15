
# 厂商信息

官网: https://www.mthreads.com/

摩尔线程智能科技（北京）有限责任公司（简称：摩尔线程）是一家以GPU芯片设计为主的集成电路设计企业，专注于研发设计全功能GPU芯片及相关产品，为科技生态合作伙伴提供强大的计算加速能力。公司致力于创新研发面向“元计算”应用的新一代GPU，构建融合视觉计算、3D图形计算、科学计算及人工智能计算的综合计算平台，建立基于云原生GPU计算的生态系统，助力驱动数字经济发展。

摩尔线程MTT  S系列全功能GPU支持多样算力，借助覆盖深度学习、图形渲染、视频处理和科学计算的完整MUSA软件栈，可为AI训练、AI推理、大模型、AIGC、云游戏、云渲染、视频云、数字孪生等场景提供通用智能算力支持，旨在为数据中心、智算中心和元计算中心的建设构建坚实算力基础，助力元宇宙中多元应用创新和落地。

MUSA软件栈通过musify CUDA代码迁移工具、计算/通信加速库、mcc编译器、musa运行时和驱动实现对CUDA生态的兼容，帮助用户快速完成代码及应用的迁移。通过torch_musa插件，可以实现MTT S系列GPU对原生PyTorch的对接，用户可以无感的把AI模型运行在摩尔线程全功能GPU上。

# FlagPerf适配验证环境说明
## 环境配置参考
  - 硬件
    - 机器型号： MCCX D800
    - 加速卡型号: MTT S3000 32GB
    - CPU型号：Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
    - 多机网络类型、带宽: InfiniBand，2*200Gbps
  - 软件
    - OS版本：Ubuntu 20.04 LTS
    - OS kernel版本: 5.4.0-154-generic
    - 加速卡驱动版本：2.2.0
    - Docker 版本: PyTorch2.0_musa1.4_ec6a747fd342 

## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：training/mthreads/docker_image/pytorch_2.0/Dockerfile
  - 构建后软件安装脚本: training/mthreads/docker_image/pytorch_2.0/pytorch_2.0_install.sh

- 核心软件信息

  - AI框架&版本
    - PyTorch: v2.0.0

  - 其它软件版本
    - torch_musa: 2.0.0+git8ea3501
    - musa toolkits: 1.4.0+git4e25703
    - mcc: 1.4.0+git5a5bcc07
    - mublas: 1.1.0+gite484aa2


## 加速卡监控采集
- 加速卡使用信息采集命令

  ```bash
  mthreads-gmi -q | grep -E 'GPU Current Temp|Power Draw|Used|Total|Gpu' | \
  awk -F ': *' '/GPU Current Temp|Power Draw|Used|Total|Gpu/ \
  { values[(NR-1)%5+1] = $2; } NR % 5 == 0 { print values[4], values[5], values[2], values[1], values[3]; }'
  ```
- 监控项示例：
    ```bash
    45C 109.51W 1MiB 32768MiB 0%
    44C 108.95W 1MiB 32768MiB 0%
    46C 110.87W 1MiB 32768MiB 0%
    43C 104.33W 1MiB 32768MiB 0%
    44C 107.55W 8MiB 32768MiB 0%
    46C 110.51W 8MiB 32768MiB 0%
    44C 106.59W 8MiB 32768MiB 0%
    44C 104.58W 8MiB 32768MiB 0%
    ```
- 加速卡使用信息采集项说明

|监控项| 日志文件 | 格式 |
|---|---|---|
|温度| mthreads_monitor.log | xxx C |
|功耗 |mthreads_monitor.log | xxx W |
|显存占用大小 |mthreads_monitor.log |xxx MiB |
|总显存大小 |mthreads_monitor.log |xxx MiB |
|显存使用率 |mthreads_monitor.log |xxx % |

