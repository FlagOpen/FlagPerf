# 厂商适配Case的规范

> 文档信息说明
> - 文档面向人群：芯片厂商开发人员
> - 文档目的：给出厂商适配标准Case的规范，降低厂商适配成本，提升共建项目的可维护性。

## 1. 厂商适配Case的代码和配置文件目录结构说明

厂商基于标准case做硬件上的适配，适配原则：
1. 优先默认使用标准case实现，不做上层接口变动，理想情况下，只有底层算子的区别，对用户不感知；
2. 接受对模型做合理优化以此来提升模型的性能表现，如bs调整等，建议底层优化，暂不接受torch接口层优化, 具体可case by case讨论。
3. 对于标准case中厂商不支持的算子，可有合理替代方案，具体可讨论。

标准Case实现路径在training/benchmarks/&lt;model&gt;/&lt;framework&gt;/下，厂商可以通过扩展模型实现的接口来适配自己的芯片。代码放在training/&lt;vendor&gt;/下，主要包括以下几部分(以Nvidia, glm, pytorch为例）：

```Bash
training/nvidia/ #厂商代码主目录
├── docker_image # 构建docker镜像所需文件
│   ├── pytorch # pytorch框架构建Docker镜像所需文件
│   │   ├── Dockerfile # 构建Docker镜像的Dockerfile
│   │   ├── packages # 如果有需要额外下载的安装包，下载至这里，并在pytorch_install.sh安装
│   │   └── pytorch_install.sh # 根据Dockerfile构建的镜像后会临时拉起镜像，运行pytorch_install.sh后commit保存镜像
├── glm-pytorch # Case适配代码和配置，目录名称格式为：<model>-<framework>
│   ├── config # 配置文件存放目录
│   │   ├── config_A100x1x2.py # 配置文件，文件名格式为：config_<卡型>X_<机器数>X<单机卡数>.py
│   │   ├── config_A100x2x8.example # 配置文件样例
│   │   ├── environment_variables.sh # 运行该Case前source该文件以配置环境
│   │   └── requirements.txt # 运行该Case前会在容器内pip安装requirements
│   ├── csrc # 算子源码目录，例如，nvidia将cuda代码放在这里，FlagPerf在运行Case前会在容器环境准备的环节调用这里的setup.py来编译安装
│   │   ├── setup.py # 算子编译安装脚本
│   └── extern # Case适配代码，可以根据基准case代码training/benchmarks/<model>/<framework>进行适配和扩展
│       ├── converter.py
│       ├── layers
│       │   ├── __init__.py
│       │   ├── layernorm.py
│       │   ├── transformer.py
│       │   └── transformer_block.py
│       └── trainer_adapter.py
├── nvidia_monitor.py # 监控脚本，在Case运行前被FlagPerf自动启动，结束后自动停止。监控脚本输出在指定的日志目录。
```

## 2. 初次适配需提供环境构建和监控程序

FlagPerf的benchmark采用Docker容器作为执行环境，并且要求在评测过程中对硬件使用率等信息进行采集。

因此，在初次适配时，需要提供构建容器镜像的Dockerfile和脚本，以及硬件监控脚本。

◊### 1）构建容器镜像

在training/下创建如下目录结构，每个AI框架一个子目录，以Nvidia+pytorch为例，需要创建training/nvidia/docker_image/pytorch/目录。

```Bash
training/<vendor>/
├── docker_image
│   ├── <framework>
```

在training/&lt;vendor&gt;/docker_image/&lt;framework&gt;下编辑Dockerfile文件，示例如下：

```Bash
FROM nvcr.io/nvidia/pytorch:21.02-py3
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c alias python3=python
```

如果依赖包安装不方便直接在写在Dockerfile里，在training/&lt;vendor&gt;/docker_image/&lt;framework&gt;下编写&lt;framework&gt;_install.sh脚本，示例如下：

```Bash
#!/bin/bash
apt-get update
apt install -y numactl
```

如果&lt;framework&gt;_install.sh脚本中需要额外下载和安装软件包，建议使用packages目录存放，并在README文档中说明。

FlagPerf会根据这里的Dockerfile及脚本自动构建容器镜像，镜像名为：flagperf-&lt;vendor&gt;-&lt;framework&gt;, tag为t_v&lt;version&gt;。出于性能考虑，FlagPerf不会在每次启动测试时构建新镜像，除非测试环境主机上不存在对应名称和tag的容器镜像。

### 2）硬件监控脚本

- FlagPerf启动Case运行的容器前，会启动系统监控信息的采集程序，在测试结束后，会结束系统监控信息的采集程序。主机CPU和内存的使用情况由FlagPerf自带的training/utils/sys_monitor.py采集。厂商需要提供自身芯片的监控信息采集脚本，放在training/&lt;vendor&gt;/目录下，要求如下：

- - 监控脚本为单一的python脚本，脚本名称&lt;vendor&gt;_monitor.py
  - 脚本支持参数：
    -  -o, --operation start|stop|restart|status
    -  -l, --log [log path] , 默认为./logs/ 
  - 支持硬件监控指标采样（必选：时间戳、使用率、显存使用率，可选：功耗、温度等，建议都有）

## 3. 厂商运行case需修改的参数

- cluster配置: training/run_benchmarks/config/cluster_conf.py

```Bash
'''Cluster configs'''

# Hosts to run the benchmark. Each item is an IP address or a hostname.
HOSTS = ["10.1.2.2", "10.1.2.3", "10.1.2.4"]  # 设置集群节点ip列表

# ssh connection port
SSH_PORT = "22" 
```

- case配置：training/run_benchmarks/config/test_conf.py

```Bash
# Set accelerator's vendor name, e.g. iluvatar, cambricon and kunlun.
# We will run benchmarks in training/&lt;vendor&gt;
VENDOR = "nvidia"   # 这里改成产商自己的名称，提交代码需复原

# Accelerator options for docker. TODO FIXME support more accelerators.
ACCE_CONTAINER_OPT = " --gpus all"   # 这里设置为空
  
# XXX_VISIBLE_DEVICE item name in env
# nvidia use CUDA_VISIBLE_DEVICE and cambricon MLU_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES for nvidia，天数
# MLU_VISIBLE_DEVICES for 寒武纪
# XPU_VISIBLE_DEVICES for 昆仑芯
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"
```

## 4. 适配FlagPerf的测试Case的要求

### 1） 组织结构

- 代码放在training/&lt;vendor&gt;/&lt;model&gt;-&lt;framework&gt;目录下，分三个目录组织：

- - config目录，存放厂商配置文件，模型在厂商机器上获得最佳性能的参数值。environment_variables.sh和requirements.txt文件用于FlagPerf在启动容器后构建环境。运行测试Case前，会先sourche environment_variables，然后使用pip安装requirements.txt中指定的包。
- - csrc目录，放算子源码和编译安装脚本setup.py。FlagPerf会在启动容器后，运行测试Case前，调用setup.py进行算子的编译和安装。
  - extern目录，模型适配的代码，标准case不支持/底层调优的相关代码放置于此。【该处会重点review】

### 2） 适配方式

模型适配代码主要通过扩展标准测试Case的接口来实现，**如遇无法通过扩展接口来支持的情况，请与FlagPerf研发团队联系。**

- 目前可扩展接口没有做严格限制，但原则上**为了保持训练workload基本一致**，要求适配过程中**不能改变**：
    - 模型结构和大小
    - 优化器类型
    - 数据集
    - 模型的初始Checkpoint

### 3）测试达标要求
- 以Perf方式训练1*1、1X8、2X8模型收敛达到目标精度(标准case中的target acc)
- 单个case的收敛时间在2-5h内
- 多机/多卡吞吐量加速比符合预期
- 支持硬件监控指标采样（必选：时间戳、使用率、显存使用率，可选：功耗、温度等，建议都有）

### 4）文档规范

按要求提供厂商的README和每个Case的README文档。文档描述需与代码实现相符合。


## 5 PR 提交规范
FlagPerf采用开源共建的方式，开发者应fork [FlagPerf仓库](https://github.com/FlagOpen/FlagPerf/tree/main) 到个人帐号下，修改代码&验证通过后，提交PR给FlagPerf项目，并指给相关reviewers。FlagOpen相关研发人员Review通过后，合并代码。具体操作可参照：https://docs.github.com/en/get-started/quickstart/contributing-to-projects

  - 只提交添加模型必要代码变动，代码格式执行"yapf -i --style "pep8" --recursive ./FlagPerf "
  - 文档齐全，case文档包括1x1, 1x8, 2x8 性能精度结果、厂商文档(如需)

