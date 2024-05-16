# 基础规格评测研发与适配文档

## 评测方案简介

为了对AI芯片这一芯片细分领域进行基础规格评测，本方案从算力、（内）存储、互联、能耗四大角度开展评测。

算力、存储、互联角度均具有若干评测项，每个评测项包含3条结果记录：**PyTorch算子或原语评测结果**、**厂商专用工具评测结果**和**厂商公布理论值**。其中：

1. **PyTorch算子或原语评测结果**：本方案仅基于PyTorch的基本算子或通信原语，实现与英伟达硬件无关的标准程序，并提供英伟达运行配置、英伟达硬件相关接口实现。**厂商需实现硬件相关接口，并提供自身运行配置**。每个评测项均会针对其所有从属配置给出调整约束，厂商可在约束内自由调整相关配置以达到更适应自身硬件特点的评测结果。**PyTorch算子或原语评测结果体现了上层用户使用时的实际情况**。
2. **厂商专用工具评测结果**：本方案引用英伟达及相关供应商-提供的二进制可执行文件或CUDA C/C++源码及Makefile，在英伟达机器上运行并获取结果。在此基础上，本方案将规定精简且信息完整的结果输出格式，并提供针对英伟达相关工具的结果处理程序（parser）。**厂商需提供自身用于评测基础规格的二进制可执行文件或C/C++级别源码及Makefile，并提供结果处理程序**，解析整理自身工具输出。**厂商专用工具评测结果体现了运行时厂商自认可合理的实际情况**。
3. **厂商公布理论值**：本方案将引用英伟达产品相关白皮书，填写所有评测项对应的理论值。**厂商在本次评测方案中可自由选择公布或保密理论值**。

能耗角度通过监控结果呈现。在上述每个评测项的每条待评测结果记录运行过程中，本方案将以固定时间间隔的方式对**AI芯片和整体服务器**进行能耗采样，形成**能耗值时间序列**。此外，本方案还将在较长时间段内采样AI芯片、整体服务器的**静默能耗**。在本方案的任何测试中，均不会由评测程序本身对功耗进行限制，只做记录。

## 工程组织形式

```
.
├── benchmarks
│   └── computation-FP32
│       ├── case_config.yaml
│       ├── main.py
│       ├── README.md
│       ├── <otherfiles>
│       └── nvidia
│           ├── case_config.yaml
│           ├── env.sh
│           ├── README.md
│           └── requirements.txt
├── configs
│   └── host.yaml
├── container_main.py
├── run.py
├── toolkits
│   └── computation-FP32
│       └── nvidia
│           ├── <otherfiles>
│           ├── main.sh
│           └── README.md
├── utils
│   ├── <otherfiles>
├── vendors
│   └── nvidia
│       ├── nvidia_analysis.py
│       ├── nvidia_monitor.py
│       └── pytorch_2.3
│           ├── Dockerfile
│           └── pytorch2.3_install.sh
```

上面的工程组织结构中，\<otherfiles\>为具体评测样例所需文件，或FlagPerf固有文件。其他文件的组织形式与FlagPerf的训练(training/)和推理(inference)相似：

1. benchmarks

   存放**PyTorch算子或原语**评测代码。每个case必定包含：

   1. case_config.yaml，为对应case的各超参配置，原则上硬件无关
   2. main.py，为对应case的主进程，由torchrun命令启动
   3. README.md，包含此case原理简单说明，并对case_config.yaml中各参数允许厂商更改的原则进行阐述
   4. vendor/目录，存放各厂商相关文件：
      1. case_config.yaml，可覆盖式更新上级目录的超参配置文件
      2. env.sh，可厂商自定义环境变量，或执行shell脚本，会在torchrun启动main.py之前由FlagPerf自动执行
      3. requirements.txt，可厂商自定义pip安装包，会在torchrun启动main.py之前由FlagPerf自动执行
      4. README.md，记录厂商此样例使用服务器的规格、芯片规格，并记录评测结果中可以公开的部分

2. configs

   下设一个文件host.yaml，存放各主机IP，端口，FlagPerf路径等信息

   此文件每次运行时自由更改填写，无需适配或更新提交

3. container_main.py

   此文件为容器内主进程，负责根据host.yaml启动对应评测样例主进程

4. run.py

   此文件为FlagPerf评测主进程，负责根据host.yaml启动并准备集群环境，启动container_main.py

5. toolkits

   存放**厂商专用工具**评测代码。每个case包含各厂商子目录，每个厂商子目录必定包含：

   1. main.sh，为对应case的主进程，由bash命令启动
   2. README.md，记录厂商此样例使用服务器的规格、芯片规格，并记录评测结果中可以公开的部分

6. utils

   包含FlagPerf所用的厂商无关工具

7. vendors

   此文件存放各厂商相关环境基础文件，每个厂商必定包含：

   1. \<vendor\>_analysis.py，用于解析各评测项结果，可参考给出的英伟达实现方案
   2. \<vendor\>_monitor.py，用于对AI芯片进行温度、功率、显存使用等方面的监控，可参考给出的英伟达实现方案
   3. \<framework\>，包含对应运行时环境

## 运行时流程

### 运行前工作

在运行评测前，需要填写configs/host.yaml

```
FLAGPERF_PATH: "/home/FlagPerf/base"
FLAGPERF_LOG_PATH: "result"
VENDOR: "nvidia"
FLAGPERF_LOG_LEVEL: "debug"
BENCHMARKS_OR_TOOLKITS: "TOOLKIT"
HOSTS: ["192.168.1.2"]
NPROC_PER_NODE: 8
SSH_PORT: "22"
HOSTS_PORTS: ["2222"]
MASTER_PORT: "29501"
SHM_SIZE: "32G"
ACCE_CONTAINER_OPT: " --gpus all"
PIP_SOURCE: "https://mirror.baidu.com/pypi/simple"
CLEAR_CACHES: True
ACCE_VISIBLE_DEVICE_ENV_NAME: "CUDA_VISIBLE_DEVICES"
CASES:
    "computation-FP32": "pytorch_2.3"

```

在host.yaml文件中，各项配置含义如下：

1. FLAGPERF_PATH，为FlagPerf/base/所在绝对路径
2. FLAGPERF_LOG_PATH，为填写日志目录相对于FlagPerf/base/的相对路径，需要具有write权限
3. VENDOR，为厂商名称
4. FLAGPERF_LOG_LEVEL，为日志记录等级，可选debug、info、error等
5. BENCHMARKS_OR_TOOLKITS，填写BENCHMARK(注意没有复数s)时表示使用**PyTorch算子或原语**评测，填写TOOLKIT时表示使用**厂商专用工具**评测。
6. HOSTS，为一个字符串数组，包含若干主机的IP。数组0位置填写的IP为MASTER
7. NPROC_PER_NODE，表示每台主机启动的AI芯片数量
8. SSH_PORT，表示主机间免密登录所用端口
9. HOST_PORTS，表示容器间torch通信所用端口
10. MASTER_PORT，表示容器间torch通信对应master端口
11. SHM_SIZE，表示容器启动时共享内存大小
12. ACCE_CONTAINER_OPT，表示AI芯片进入镜像所需命令。例如对于英伟达，命令为" --gpus all"
13. PIP_SOURCE，表示容器内PIP所用源地址
14. CLEAR_CACHE，表示启动测试前是否清理系统cache，原则上为True
15. ACCE_VISIBLE_DEVICE_ENV_NAME，表示选定AI芯片所用环境变量。例如对于英伟达，环境变量为"CUDA_VISIBLE_DEVICES"
16. CASES，为一个字典。key为评测样例名称，value为对应运行时环境名称

### 运行流程

* 此运行流程为FlagPerf自动进行，在此仅供研发人员调试参考，不需手动执行。

1. 在FlagPerf/base/执行python3 run.py【此步骤需手动执行】
2. run.py启动并准备好集群容器环境
3. run.py在每一个主机的物理机环境启动监控
4. run.py在每一个主机的容器内自动启动container_main.py并自动给定所需命令行参数
5. container_main.py根据配置，执行torchrun benchmarks/....../main.py --args或bash toolkits/....../main.sh启动评测任务
6. 容器内评测任务结束后，run.py关闭所有运行时容器，关闭监控
7. run.py将所有主机的log文件复制到master节点
8. run.py调用各厂商提供analysis.py文件，获取规格化结果
9. run.py将评测指标结果打印到标准输出，将详细规格化结果以json形式保存至master节点的log目录

## 厂商适配文档

### 初次适配

厂商初次参与某样例评测时，需要提交：

1. 适配样例的case_config.yaml，env.sh，requirements.txt，根据机器信息及评测结果填写README.md，位于benchmarks/\<case\>/\<vendor\>/
2. 提交样例的main.sh及其他可能的文件，根据机器信息及评测结果填写README.md，位于toolkits/\<case\>/\<vendor\>/
3. 提交厂商自身相关环境文件及相关代码，即vendors/\<vendor\>/目录，组织形式及内容可参考英伟达方案

### 后续适配

厂商后续参与某评测样例时，需要提交初次适配所需的1、2两部分文件，不需要提交第3部分

### 配置及结果更新

厂商如认为现有评测结果及配置不足以展现自身能力，可修改初次适配所需的1、2两部分文件，修改两个README.md中的结果。

### 环境或硬件更新

厂商如需对vendors/\<vendor\>/目录下文件进行更新，或采用新款芯片进行评测，需在下列方式中选择一种：

1. 更新vendors/\<vendor\>/目录下文件，补充所有已适配case在新版vendors/\<vendor\>/下的评测结果
2. 新增vendors/\<vendor_suffix\>/目录，提交第3部分文件。针对要使用新版环境或硬件评测的样例，单独提交benchmarks/\<case\>/\<vendor_suffix\>/ 及 toolkits/\<case\>/\<vendor_suffix\>/
