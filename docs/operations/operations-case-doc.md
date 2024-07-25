# 算子评测厂商适配文档

为了评估AI芯片在原生算子和Triton算子（[FlagGems](https://github.com/FlagOpen/FlagGems)）方面的支持程度和性能，FlagPerf 设计并实现了针对各个算子在 AI 芯片的评测方案。具体的评测方案细节可以向 FlagPerf 团队索取评测方案文档，这里仅介绍厂商适配需关注的详细内容。厂商按照本文档完成适配后，评测时会自动生成相关指标结果。

## 工程组织形式
算子评测相关代码均在FlagPerf/operation目录下, 整体结构如下：
```
├── benchmarks
│   ├── abs
│   │   ├── case_config.yaml
│   │   ├── main.py
│   │   └── nvidia
│   │       └── A100_40_SXM
│   │           ├── README.md
│   │           ├── case_config.yaml
│   │           ├── env.sh
│   │           └── requirements.txt
├── configs
│   └── host.yaml
├── container_main.py
├── run.py
└── vendors
    └── nvidia
        ├── ngctorch2403
        │   ├── Dockerfile
        │   └── ngctorch2403_install.sh
        ├── nvidia_analysis.py
        └── nvidia_monitor.py
        
```
1、benchmarks

存放各个算子评测代码。每个算子必定包含：

* case_config.yaml，为对应算子的各超参配置，原则上硬件无关
* main.py，为对应算子的主进程
* vendor/目录，存放各厂商相关文件：
    * case_config.yaml，可覆盖式更新上级目录的超参配置。原则上推荐采用FlagPerf 的默认配置，如果因对应芯片无法支持FlagPerf默认配置, 可以在该文件中修改超参配置
    * env.sh，可厂商自定义针对该算子的环境变量/执行shell脚本，会在启动main.py之前由FlagPerf自动执行
    * requirements.txt，可厂商自定义pip安装包，会由FlagPerf自动执行
    * README.md，记录厂商此样例使用服务器的规格、芯片规格，并记录评测结果中可以公开的部分

2、configs
下设一个文件host.yaml，存放各主机IP，端口，FlagPerf路径等信息

此文件每次运行时自由更改填写，无需适配或更新提交

3、container_main.py

此文件为容器内主进程，负责根据host.yaml启动对应评测样例主进程

4、run.py

此文件为FlagPerf评测主进程，负责根据host.yaml启动并准备集群环境，启动container_main.py

5、vendors

此文件存放各厂商相关环境基础文件，每个厂商必定包含：
*  \<vendor\>_analysis.py，用于解析各评测项结果，可参考给出的英伟达实现方案
*  \<vendor\>_monitor.py，用于对AI芯片进行温度、功率、显存使用等方面的监控，可参考给出的英伟达实现方案
*  \<envname\>，包含对应运行时环境
    *  Dockerfile :
    *  \<envname\>_install.sh: 可自定义全局的环境变量、安装软件等操作，会在测例评测开始由FlagPerf 自动执行

注意: 这里的**envname**需要和host.yaml中 CASES 的 value 值保持一致。可以参考下图英伟达的命名与使用方式。
![sample](assets/sample.jpg)


## 评测运行时流程

#### 运行前工作
1、配置修改
在运行评测前，需要填写configs/host.yaml文件
```
FLAGPERF_PATH: "/home/FlagPerf/operation"
FLAGPERF_LOG_PATH: "result"
VENDOR: "nvidia"
FLAGPERF_LOG_LEVEL: "info"
HOSTS: ["192.168.1.2"]
NPROC_PER_NODE: 1
SSH_PORT: "22"
HOSTS_PORTS: ["2222"]
MASTER_PORT: "29501"
SHM_SIZE: "32G"
ACCE_CONTAINER_OPT: " --gpus all"
# for nvidia, using " -- gpus all"
# for xxx, using
PIP_SOURCE: "https://mirror.baidu.com/pypi/simple"
CLEAR_CACHES: True
# for nvidia, using "CUDA_VISIBLE_DEVICES"
# for xxx, using
ACCE_VISIBLE_DEVICE_ENV_NAME: "CUDA_VISIBLE_DEVICES"
# "operation:dataFormat:chip": "docker_images"
# now only support flaggems and nativepytorch
CASES: 
    "mm:FP16:nativetorch:A100_40_SXM": "ngctorch2403"
```
在host.yaml文件中，各项配置含义如下：

* FLAGPERF_PATH: 为FlagPerf/operation/所在**绝对路径**
* FLAGPERF_LOG_PATH: 为填写日志目录相对于FlagPerf/operation/的**相对路径**，需要具有write权限
* VENDOR: 为厂商名称
* FLAGPERF_LOG_LEVEL: 为日志记录等级，可选debug、info、error等
* HOSTS:为一个字符串数组，包含若干主机的IP。数组0位置填写的IP为MASTER
* NPROC_PER_NODE: 表示每台主机启动的AI芯片数量
* SSH_PORT: 表示主机间免密登录所用端口
* HOST_PORTS: 表示容器间torch通信所用端口
* MASTER_PORT: 表示容器间torch通信对应master端口
* SHM_SIZE:表示容器启动时共享内存大小
* ACCE_CONTAINER_OPT: 表示AI芯片进入镜像所需命令。例如对于英伟达，命令为" --gpus all"
* PIP_SOURCE: 表示容器内PIP所用源地址
* CLEAR_CACHE: 表示启动测试前是否清理系统cache，原则上为True
* ACCE_VISIBLE_DEVICE_ENV_NAME: 表示选定AI芯片所用环境变量。例如对于英伟达，环境变量为"CUDA_VISIBLE_DEVICES"
* CASES: 为一个字典。key为评测算子名称:数制:算子库名:芯片型号, value为对应运行时环境名称。
    例如，可使用"mm:FP16:nativetorch:A100_40_SXM": "ngctorch2403" 来以FP6数制基于原生NativeTorch执行mm算子；
    可使用"mm:FP32:flaggems:A100_40_SXM": "ngctorch2403" 来以FP32数制基于FlagGems算子库执行mm算子；
    ngctorch2403 为vendors目录下被评测厂商对应运行环境的名称。

2、运行流程
为了更好的理解整体流程，这里以流程图的形式简述单个算子评测的主要流程。
![单个算子执行流程](assets/%E5%8D%95%E4%B8%AA%E7%AE%97%E5%AD%90%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B.png)

3、快速评测方法
因算子数量众多，本方案提供了快速执行所有算子并渲染结果的脚本，以帮助厂商快速确认。脚本位于``` operation/helper``` 目录下，使用方法如下：
```
# 安装依赖
cd operation/helper
pip install -r requirements.txt

# 按照实际情况修改 main.sh 脚本
vim main.sh 
# 该脚本中有两处需要修改
# （1）与厂商、执行环境相关的数据，即“修改点 1”
# （2）与测例相关的数据，即“修改点 2”，其格式为 算子名="数制1 数制2 数制3"

# 执行 main.sh 即可, 执行完成后会在operation/results 目录下看到每个算子、每个数制的执行结果和日志
bash main.sh
```


## 厂商适配文档
#### 初次适配
如“评测运行时流程”的“运行流程”中**黄颜色部分**所示，厂商需要适配的分为三个部分：
* 适配样例的case_config.yaml，env.sh，requirements.txt，根据机器信息及评测结果填写README.md，位于benchmarks/\<case\>/\<vendor>\/, 可参考英伟达方案
* 适配监控和日志分析等方法，该部分位于vendors/\<vendor\>/目录下，形式与内容可以参考英伟达方案
* 提交厂商自身相关环境文件及相关代码，即vendors/\<vendor\>/\<环境名\>目录，组织形式及内容可参考英伟达方案
#### 后续适配
厂商后续参与某评测样例时，需要提交初次适配所需的 1 部分文件，不需要提交第2、3部分
#### 配置及结果更新
厂商如认为现有评测结果及配置不足以展现自身能力，可修改初次适配所需的 1 部分中的文件，并修改README.md中的结果。
