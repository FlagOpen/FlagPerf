### 部署说明

#### 软件环境

```
    OS: Ubuntu 20.04
    Kernel: 5.4.0-52-generic
    Docker: 20.10.9
    Python: 3.8
```

#### 代码目录说明


```
├── LICENSE.md # 版权信息
├── README.md  # 本文件
└── training
    ├── benchmarks  # benchmark的标准实现
    ├── nvidia      # 厂商配置及扩展
    ├── requirements.txt # FlagPerf依赖的python包
    ├── run_benchmarks # 测试任务的脚本和配置
    └── utils # 测试任务执行需要的工具
```

#### 下载FlagPerf并部署

1）在测试集群所有服务器上执行以下命令

```
# git clone https://github.com/FlagOpen/FlagPerf.git  
# cd FlagPerf/training/
# pip3 install -r requirements.txt
```

2）配置集群各服务器间root帐号的ssh信任关系和sudo免密

----------

### 快速启动

#### 准备数据和模型checkpoint

在测试集群的每台服务器上，准备数据和模型checkpoint，参见benchmarks/\<model\>/\<framework\>/README.md

#### 编辑配置文件

在准备发起测试任务的服务器上，修改配置文件。主要包括集群配置文件、测试配置文件和benchmark模型配置文件。

1）修改集群配置文件

集群配置文件在FlagPerf/training/run_benchmarks/config/cluster_conf.py

```
# cd Flagperf/training/
# vim run_benchmarks/config/cluster_conf.py
```

集群配置文件主要包括集群主机列表和SSH端口，示例如下：

```
'''Cluster configs'''

# Hosts to run the benchmark. Each item is an IP address or a hostname.
HOSTS = ["10.1.2.3", "10.1.2.4", "10.1.2.5", "10.1.2.6"]
# ssh connection port
SSH_PORT = "22"
```

2）修改测试配置文件
测试配置文件在FlagPerf/training/run_benchmarks/config/test_conf.py，主要包括FlagPerf的部署路径、数据和模型checkpoint的路径、要跑的测试benchmark case列表等。

__Tips：__

 * 请根据自己所在地区，选用合适的pip源来配置PIP_SOURCE
 * 每次运行可配置多个benchmark case，每个benchmark case可以通过repeat来配置运行次数
 * FlagPerf使用CASES变量中的键（key）来索引相应模型（model，如bert），框架（framework，可选pytorch、pytorch_1.13），硬件类型（hardware_model，如A100）,主机数量（nnodes，如1），计算卡数量（nproc，如8），和重复测试次数（repeat，如1），以冒号:为分隔符，按照“model:framework:hardware_model:nnodes:nproc:repeat”的格式以字符串存储。键对应的值为运行这一样例对应数据/模型权重所在目录
 * 例如，用户在目录/abc/def/data/存放了模型bert在框架pytorch下面运行的数据集与预训练权重，希望在2机8卡A100（共16卡）的环境上测试这一任务，重复3次取平均值，则需要在CASES中增加"bert:pytorch:A100:2:8:3":"/abc/def/data/"这一键值对。key中的bert为模型，pytorch为框架，A100为硬件类型，2为主机数量，8为每个主机上面的计算卡数量，3为重复次数，"abc/def/data/"为数据和权重的存放路径

```
'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon, kunlunxin and ascend.
# We will run benchmarks in training/<vendor>
VENDOR = "nvidia"

# Accelerator options for docker. TODO FIXME support more accelerators.
# possible value of ACCE_CONTAINER_OPT are:
#   iluvatar:
#       ' -v /lib/modules:/lib/modules '
#   kunlunxin:
#       " --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2" + \
#       " --device=/dev/xpu3 --device=/dev/xpu4 --device=/dev/xpu5" + \
#       " --device=/dev/xpu6 --device=/dev/xpu7 --device=/dev/xpuctrl"
#   nvidia:
#       " --gpus all"
#   ascend:
#       "--device=/dev/davinciX --device=/dev/davinci_manager + \
#        --device=/dev/devmm_svm --device=/dev/hisi_hdc + \
#        -v /usr/local/Ascend/driver -v /usr/local/dcmi -v /usr/local/bin/npu-smi"
ACCE_CONTAINER_OPT = " --gpus all"
# XXX_VISIBLE_DEVICE item name in env
# possible value of ACCE_VISIBLE_DEVICE_ENV_NAME are:
#   CUDA_VISIBLE_DEVICES for nvidia, iluvatar
#   MLU_VISIBLE_DEVICES for cambricon
#   XPU_VISIBLE_DEVICES for kunlunxin
#   ASCEND_VISIBLE_DEVICES for ascend
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set pip source, which will be used in preparing envs in container
PIP_SOURCE = "https://mirror.baidu.com/pypi/simple"

# The path that flagperf deploy in the cluster.
# Users must set FLAGPERF_PATH to where flagperf deploy
# You can assume the preset "/home/FlagPerf/training" points to Null
FLAGPERF_PATH = "/home/FlagPerf/training"
# Set log path on the host here.
FLAGPERF_LOG_PATH = FLAGPERF_PATH + "/result/"

# Set log level. It should be 'debug', 'info', 'warning', or 'error'.
FLAGPERF_LOG_LEVEL = 'debug'

# System config
# Share memory size
SHM_SIZE = "32G"
# Clear cache config. Clean system cache before running testcase.
CLEAR_CACHES = True

# Set the case dict you want to run here.
'''
# Users must use {
    "model:framework:hardwareID:nnodes:nproc:repeat": "dataset path"}
'''
CASES = {
    "bert:pytorch_1.8:A100:1:8:1": "/home/datasets_ckpt/bert/train/",
    "glm:pytorch_1.8:A100:1:8:1": "/home/datasets_ckpt/glm/train/"
}

```

3）修改Vendor目录下的benchmark case配置文件（视自身需求，也可不修改）  
benchmark case配置文件在\<vendor\>/\<modle\>-\<framework\>/config目录下，例如nvidia/bert-paddle/config/config_A100x1x8.py，示例如下：

```
target_mlm_accuracy = 0.67
gradient_accumulation_steps = 1
max_steps = 10000
start_warmup_step = 0
warmup_proportion = 0
warmup_steps = 2000

learning_rate = 1e-4
weight_decay_rate = 0.01
opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999
train_batch_size = 12
eval_batch_size = train_batch_size
max_samples_termination = 4500000
cache_eval_data = False

seed = 9031
```

#### 启动测试

一条命令即可启动一组测试，在修改好配置的服务器上，进入training目录，运行如下命令

```
# python3 ./run_benchmarks/run.py
```

可看到测试执行过程，为了防止断网等情况 ，推荐使用`nohup python3 ./run_benchmarks/run.py &`启动。输出如下：

```
==============================================
          Welcome to FlagPerf!
      See more at https://github.com/FlagOpen/FlagPerf 
==============================================
2022-11-21 19:19:24,013	[INFO]	[run.py,500]======== Step 1: Check environment and configs. ========
2022-11-21 19:19:24,014	[INFO]	[run.py,501]Initialize logger with log path: /home/FlagPerf/training/result/run20221121191924......[SUCCESS]
2022-11-21 19:19:24,014	[DEBUG]	[run.py,38]Cluster healthcheck ssh. Hosts are: 10.1.2.2
2022-11-21 19:19:24,014	[DEBUG]	[cluster_manager.py,43]Run cmd on host with ssh. ssh cmd=ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -l root -p 22 10.1.2.2 ':' host=10.1.2.2 timeout=10
2022-11-21 19:19:24,997	[INFO]	[run.py,47]Check hosts in the cluster......[SUCCESS]
2022-11-21 19:19:24,997	[DEBUG]	[run.py,63]Check flagperf deployment path: /home/FlagPerf/training
2022-11-21 19:19:24,997	[DEBUG]	[cluster_manager.py,43]Run cmd on host with ssh. ssh cmd=ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -l root -p 22 10.1.2.2 'cd /home/FlagPerf/training' host=10.1.2.2 timeout=10
2022-11-21 19:19:25,780	[INFO]	[run.py,71]Check flagperf deployment path: /home/FlagPerf/training...[SUCCESS]
2022-11-21 19:19:25,780	[DEBUG]	[run.py,79]Check test config: VENDOR
2022-11-21 19:19:25,780	[INFO]	[run.py,90]Check test config: VENDOR......[SUCCESS]
2022-11-21 19:19:25,780	[DEBUG]	[run.py,420]Check configs of all test cases: GLM_TORCH_DEMO_A100_1X8,CPM_TORCH_DEMO_A100_1X8
2022-11-21 19:19:25,780	[DEBUG]	[run.py,97]Check config of test case: GLM_TORCH_DEMO_A100_1X8

......中间日志省略......

2022-11-21 20:36:19,554	[DEBUG]	[cluster_manager.py,43]Run cmd on host with ssh. ssh cmd=ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -l root -p 22 10.1.2.2 'cd /home/FlagPerf/training && /usr/bin/python3 /home/FlagPerf/training/nvidia/nvidia_monitor.py -o stop' host=10.8.200.155 timeout=60
2022-11-21 20:36:21,400	[INFO]	[run.py,583]-== Testcase CPM_TORCH_DEMO_A100_1X8 Round 1 finished ==-
2022-11-21 20:36:21,401	[INFO]	[run.py,585]=== 2.3 Setup container and run testcases finished. ===
2022-11-21 20:36:21,401	[INFO]	[run.py,587]========= Step 3: Collect logs in the cluster. =========
2022-11-21 20:36:21,401	[INFO]	[run.py,388]Collect logs in cluster.
2022-11-21 20:36:21,401	[DEBUG]	[run.py,394]Case GLM_TORCH_DEMO_A100_1X8, round 1, log dir: /home/FlagPerf/training/result/run20221121191924/GLM_TORCH_DEMO_A100_1X8/round1
2022-11-21 20:36:21,401	[DEBUG]	[cluster_manager.py,164]scp command:scp -o  ConnectTimeout=3 -o StrictHostKeyChecking=no -P 22 -r root@10.1.2.2:/home/FlagPerf/training/result/run20221121191924/* /home/FlagPerf/training/result/run20221121191924/
2022-11-21 20:36:22,332	[INFO]	[run.py,408]Case GLM_TORCH_DEMO_A100_1X8, round 1, get all logs in dir: /home/FlagPerf/training/result/run20221121191924/GLM_TORCH_DEMO_A100_1X8/round1
2022-11-21 20:36:22,332	[DEBUG]	[run.py,394]Case CPM_TORCH_DEMO_A100_1X8, round 1, log dir: /home/FlagPerf/training/result/run20221121191924/CPM_TORCH_DEMO_A100_1X8/round1
2022-11-21 20:36:22,332	[DEBUG]	[cluster_manager.py,164]scp command:scp -o  ConnectTimeout=3 -o StrictHostKeyChecking=no -P 22 -r root@10.1.2.2:/home/FlagPerf/training/result/run20221121191924/* /home/FlagPerf/training/result/run20221121191924/
2022-11-21 20:36:23,239	[INFO]	[run.py,408]Case CPM_TORCH_DEMO_A100_1X8, round 1, get all logs in dir: /home/FlagPerf/training/result/run20221121191924/CPM_TORCH_DEMO_A100_1X8/round1
2022-11-21 20:36:23,239	[INFO]	[run.py,412]Congrats! See all logs in /home/FlagPerf/training/result/run20221121191924
2022-11-21 20:36:23,239	[INFO]	[run.py,595]Stop FlagperfLogger.
```

PS：执行过程中，长时间处于 `2.1 Prepare docker image:` 状态无log输出属于正常现象，底层执行镜像拉取操作，请耐心等待镜像拉取完成或在主机预装对应镜像。

#### 查看日志

日志在配置的日志目录里run\<timestamp\>目录下，每个<benchmark case name>一个子目录，每轮测试放在round\<X\>子目录下，每个node有一个目录，放置其中每个rank的日志和cpu、内存等系统监控。例如：

```
# cd result/run20221121191924/CPM_TORCH_DEMO_A100_1X8/
# ls
round1
# ls round1/
10.1.2.2_noderank0
# cd 10.1.2.2_noderank0/
# ls
cpu_monitor.log     pwr_monitor.log  rank2.out.log  rank5.out.log  start_pytorch_task.log
mem_monitor.log     rank0.out.log    rank3.out.log  rank6.out.log
nvidia_monitor.log  rank1.out.log    rank4.out.log  rank7.out.log
```

以pytorch的benchmark case为例，rank0可以看到训练结果和日志。

```
# tail -n 6 rank0.out.log
[PerfLog] {"event": "STEP_END", "value": {"loss": 2.679504871368408, "embedding_average": 0.916015625, "epoch": 1, "end_training": true, "global_steps": 3397, "num_trained_samples": 869632, "learning_rate": 0.000175375, "seq/s": 822.455385237589}, "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 127, "time_ms": 1669034171032, "rank": 0}}
[PerfLog] {"event": "EVALUATE", "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 127, "time_ms": 1669034171032, "rank": 0}}
[PerfLog] {"event": "EPOCH_END", "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 127, "time_ms": 1669034171159, "rank": 0}}
[PerfLog] {"event": "TRAIN_END", "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 136, "time_ms": 1669034171159, "rank": 0}}
[PerfLog] {"event": "FINISHED", "value": {"e2e_time": 1661.6114165782928, "training_sequences_per_second": 579.0933420700227, "converged": true, "final_loss": 3.066718101501465, "final_mlm_accuracy": 0.920166015625, "raw_train_time": 1501.713, "init_time": 148.937}, "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 158, "time_ms": 1669034171646, "rank": 0}}

```


> 说明：
> \<IP\>_noderank\<X\> ：训练日志 noderank为\<X\>的节点日志
> cpu_monitor.log：训练过程中的CPU监控日志。格式：采样时间点，平均使用率
> gpu_monitor.log：训练过程中的GPU监控日志。格式：采样时间点，每行包括：卡X温度，卡X功率，卡X显存使用，卡X显存大小，卡X使用率
> mem_monitor.log：训练过程中的内存监控日志。格式：采样时间点，平均使用率
> pwr_monitor.log：训练过程中的电源监控日志。格式：采样时间点，整机功率

----------

### 