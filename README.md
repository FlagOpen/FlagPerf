![FlagAI](logo.png)

### FlagPerf

FlagPerf是一款面向AI异构加速芯片的通用基准性能测试平台。我们希望面向AI异构芯片提供多框架、多场景、开放、公平的标准化测试体系，通过抽象模型训练过程及厂商共建扩展的形式，兼顾AI性能测试的通用性和异构硬件的灵活性。
更多模型及框架支持正在持续开发中，欢迎加入共同建设，助力异构芯片产业生态发展。
### 安装部署

系统环境建议


```
    OS：Ubuntu 20.04
    Kernel：5.4.0-52-generic
    Docker：20.10.9
    Docker OS：Ubuntu 18.04
    Python：3.8
```

代码目录说明


```
├── LICENSE.md # 版权信息
├── README.md  # 本文件
└── training
    ├── benchmarks  # 测试Case的标准实现
    ├── nvidia      # 厂商配置及扩展
    ├── requirements.txt # FlagPerf依赖的python包
    ├── run_benchmarks # 测试任务的脚本和配置
    └── utils # 测试任务执行需要胡工具
```

下载并部署

```
在测试环境所有服务器上执行以下命令：
git clone https://github.com/FlagOpen/FlagPerf.git  
cd flagperf
cd training
pip3 install -r requirements.txt
配置服务器间root帐号的ssh信任关系
```

### 快速启动

#### 准备数据和模型checkpoint

参见benchmarks/<model>目录/README.md
#### 编辑配置文件

修改集群配置文件

```
cd flagperf/training/
vim run_benchmarks/config/cluster_conf.py
'''Cluster configs'''

# Hosts to run the benchmark. Each item is an IP address or a hostname.
HOSTS = ["10.1.2.3", "10.1.2.4", "10.1.2.5", "10.1.2.6"]
# ssh connection port
SSH_PORT = "22"
```

修改测试配置文件

```
vim run_benchmarks/config/test_conf.py
'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon and kunlun.
# We will run benchmarks in training/<vendor>
VENDOR = "nvidia"
# Accelerator options for docker. TODO FIXME support more accelerators.
ACCE_CONTAINER_OPT = " --gpus all"
# XXX_VISIBLE_DEVICE item name in env
# nvidia use CUDA_VISIBLE_DEVICE and cambricon MLU_VISIBLE_DEVICES
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set type of benchmarks, default or customized.
# default: run benchmarks in training/benchmarks/
# [NOT SUPPORTED] customized: run benchmarks in training/<vendor>/benchmarks/
TEST_TYPE = "default"

# The path that flagperf deploy in the cluster.
# If not set, it will be os.path.dirname(run.py)/../../training/
FLAGPERF_PATH_HOST = "/home/flagperf/training"

# Set the mapping directory of flagperf in container.
FLAGPERF_PATH_CONTAINER = "/workspace/flagperf/training"

# Set log path on the host here. TODO use another path in example.
FLAGPERF_LOG_PATH_HOST = FLAGPERF_PATH_HOST + "/result/"
# Set log path in container here.
FLAGPERF_LOG_PATH_CONTAINER = FLAGPERF_PATH_CONTAINER + "/result/"
# Set log level. It should be 'debug', 'info', 'warning' or 'error'.
FLAGPERF_LOG_LEVEL = 'debug'

# System config
# Share memory size
SHM_SIZE = "32G"
# Clear cache config. Clean system cache before running testcase.
CLEAR_CACHES = True

# Set cases you want to run here.
# cases is a list of case name.
CASES = ['BERT_TORCH_DEMO_A100_1X8']

# Config each case in a dictionary like these.
# <case name> = {
#     # "Set model name"
#     "model": <model name>
#     # If test_type is default, framework should be pytorch.
#     "framework": "<ai framework>",
#     # Set config module in <vendor>/<model>-<framework>/<config>
#     "config": "<testcase config module>",
#     # Set how many times to run this case in container(s).
#     "repeat": 1,
#     # Set how many hosts to run this case
#     "nnodes": 1,
#     # Set how many processes will run on each host
#     "nproc": 2,
#     # Set data path on host: "/home/data_ckpt/bert/train"
#     "data_dir_host": "<data direcotory on host>",
#     # Set data path in container: /mnt/data/bert/train"
#     "data_dir_container": "<data direcotory in container>",
# }

BERT_PADDLE_DEMO_A100_1X8 = {
    "model": "bert",
    "framework": "paddle",
    "config": "config_A100x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/bert/train/",
    "data_dir_container": "/mnt/data/bert/train/",
}
```

修改Vendor的测试Case配置文件

```
vim nvidia/bert-paddle/config/config_A100x1x8.py
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


```
# python3 ./run_benchmarks/run.py
==============================================
          Welcome to flagperf!
      See more at https://baai.ac.cn/ 
==============================================
2022-09-28 16:10:33,603    [INFO]    [run.py,481]======== Step 1: Check environment and configs. ========
2022-09-28 16:10:33,603    [INFO]    [run.py,483]Initialize logger with log path: /home/flagperf/training/result/run20220928161033......[SUCCESS]
2022-09-28 16:10:33,603    [DEBUG]    [run.py,36]Cluster healthcheck ssh. Hosts are: 10.1.2.3
2022-09-28 16:10:33,603    [DEBUG]    [cluster_manager.py,42]Run cmd on host with ssh. ssh cmd=ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -l root -p 22 10.1.2.3 ':' host=10.1.2.3 timeout=10
2022-09-28 16:10:34,764    [INFO]    [run.py,44]Check hosts in the cluster......[SUCCESS]
2022-09-28 16:10:34,764    [DEBUG]    [run.py,60]Check flagperf deployment path: /home/flagperf/training
2022-09-28 16:10:34,764    [DEBUG]    [cluster_manager.py,42]Run cmd on host with ssh. ssh cmd=ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -l root -p 22 10.1.2.3 'cd /home/flagperf/training' host=10.1.2.3 timeout=10
2022-09-28 16:10:35,732    [INFO]    [run.py,69]Check flagperf deployment path: /home/flagperf/training...[SUCCESS]
2022-09-28 16:10:35,732    [DEBUG]    [run.py,76]Check test config: TEST_TYPE and VENDOR
2022-09-28 16:10:35,732    [INFO]    [run.py,87]Check test config: TEST_TYPE and VENDOR......[SUCCESS]
2022-09-28 16:10:35,732    [DEBUG]    [run.py,415]Check configs of all test cases: BERT_TORCH_DEMO_A100_1X8
2022-09-28 16:10:35,732    [DEBUG]    [run.py,94]Check config of test case: BERT_TORCH_DEMO_A100_1X8
2022-09-28 16:10:35,732    [DEBUG]    [run.py,133]Check config of test case: BERT_TORCH_DEMO_A100_1X8 ...[SUCCESS]
2022-09-28 16:10:35,732    [DEBUG]    [run.py,431]Valid cases: BERT_TORCH_DEMO_A100_1X8
2022-09-28 16:10:35,732    [DEBUG]    [run.py,433]Invalid cases that can't find config: 
2022-09-28 16:10:35,732    [DEBUG]    [run.py,435]Invalid cases that config is error: 
2022-09-28 16:10:35,732    [INFO]    [run.py,436]Get valid cases list......[SUCCESS]
2022-09-28 16:10:35,732    [INFO]    [run.py,459]--------------------------------------------------
2022-09-28 16:10:35,732    [INFO]    [run.py,460]Prepare to run flagperf benchmakrs with configs: 
2022-09-28 16:10:35,733    [INFO]    [run.py,461]Deploy path on host:    /home/flagperf/training
2022-09-28 16:10:35,733    [INFO]    [run.py,462]Vendor:    nvidia
2022-09-28 16:10:35,733    [INFO]    [run.py,463]Test type:    default
2022-09-28 16:10:35,733    [INFO]    [run.py,464]Testcases:    [BERT_TORCH_DEMO_A100_1X8]
2022-09-28 16:10:35,733    [INFO]    [run.py,465]Log path on host:    /home/flagperf/training/result/run20220928161033
2022-09-28 16:10:35,733    [INFO]    [run.py,466]Cluster:    [10.1.2.3]
2022-09-28 16:10:35,733    [INFO]    [run.py,467]--------------------------------------------------
2022-09-28 16:10:35,733    [INFO]    [run.py,495]========= Step 2: Prepare and Run test cases. =========
2022-09-28 16:10:35,733    [INFO]    [run.py,498]======= Testcase: BERT_TORCH_DEMO_A100_1X8 =======
2022-09-28 16:10:35,733    [INFO]    [run.py,507]=== 2.1 Prepare docker image:flagperf-nvidia-paddle:t_v0.1 ===
......中间日志省略......
2022-09-28 16:16:00,924    [INFO]    [run.py,563]========= Step 3: Collect logs in the cluster. =========
2022-09-28 16:16:00,924    [INFO]    [run.py,383]Collect logs in cluster.
2022-09-28 16:16:00,925    [DEBUG]    [run.py,390]Case BERT_TORCH_DEMO_A100_1X8, round 1, log dir: /home/flagperf/training/result/run20220928161033/BERT_TORCH_DEMO_A100_1X8/round1
2022-09-28 16:16:00,925    [DEBUG]    [cluster_manager.py,113]scp command:scp -o  ConnectTimeout=3 -o StrictHostKeyChecking=no -P 22 -r root@10.1.2.3:/home/flagperf/training/result/run20220928161033/* /home/flagperf/training/result/run20220928161033/
2022-09-28 16:16:01,941    [INFO]    [run.py,404]Case BERT_TORCH_DEMO_A100_1X8, round 1, get all logs in dir: /home/flagperf/training/result/run20220928161033/BERT_TORCH_DEMO_A100_1X8/round1
2022-09-28 16:16:01,941    [INFO]    [run.py,407]Congrats! See all logs in /home/flagperf/training/result/run20220928161033
2022-09-28 16:16:01,941    [INFO]    [run.py,571]Stop FlagperfLogger.
```


#### 查看日志


```
cd result/run<timpstamp>/BERT_TORCH_DEMO_A100_1X8/round<X>/
ls
10.1.2.3_noderank0

cd 10.1.2.3_noderank0/
ls
cpu_monitor.log  pwr_monitor.log  rank2.out.log  rank5.out.log  start_paddle_task.log
gpu_monitor.log  rank0.out.log    rank3.out.log  rank6.out.log
mem_monitor.log  rank1.out.log    rank4.out.log  rank7.out.log

tail -n 5 rank<Y>.out.log # 可以看到rank<Y>训练过程和结果日志。
[PerfLog] {"event": "STEP_END", "value": {"loss": 1.298828125, "mlm_acc": 0.7184170484542847, "epoch": 0, "end_training": true, "num_trained_samples": 10500864, "global_steps": 41020, "iter_dataloader_idx": 8, "learning_rate": 0.00020643, "seq/s": 1302.1523714815849}, "metadata": {"file": "/workspace/flagperf/training/benchmarks/bert/paddle/run_pretraining.py", "lineno": 137, "time_ms": 1665310732959, "rank": 0}}
[PerfLog] {"event": "EVALUATE", "metadata": {"file": "/workspace/flagperf/training/benchmarks/bert/paddle/run_pretraining.py", "lineno": 137, "time_ms": 1665310732960, "rank": 0}}
[PerfLog] {"event": "EPOCH_END", "metadata": {"file": "/workspace/flagperf/training/benchmarks/bert/paddle/run_pretraining.py", "lineno": 137, "time_ms": 1665310733109, "rank": 0}}
[PerfLog] {"event": "TRAIN_END", "metadata": {"file": "/workspace/flagperf/training/benchmarks/bert/paddle/run_pretraining.py", "lineno": 147, "time_ms": 1665310733110, "rank": 0}}
[PerfLog] {"event": "FINISHED", "value": {"e2e_time": 8649.404755353928, "training_sequences_per_second": 1221.242941532543, "converged": true, "final_loss": 1.3019317388534546, "final_mlm_accuracy": 0.7202467322349548, "raw_train_time": 8598.715, "init_time": 40.799}, "metadata": {"file": "/workspace/flagperf/training/benchmarks/bert/paddle/run_pretraining.py", "lineno": 168, "time_ms": 1665310733121, "rank": 0}}

```


> 说明：
> \<IP\>_noderank\<X\> ：训练日志 noderank为\<X\>的节点日志
> cpu_monitor.log：训练过程中的CPU监控日志。格式：采样时间点，平均使用率
> gpu_monitor.log：训练过程中的GPU监控日志。格式：采样时间点，卡1温度，卡1功率，卡1显存使用，卡1使用率……
> mem_monitor.log：训练过程中的内存监控日志。格式：采样时间点，平均使用率
> pwr_monitor.log：训练过程中的电源监控日志。格式：采样时间点，整机功率

### 教程


- BERT-Large
- GLM-Large
- CPM-1-medium

### 贡献代码

本项目目前由北京智源人工智能研究院、天数智芯、与百度PaddlePaddle共同建设中。
诚邀各框架、芯片团队与个人参与！如果您兴趣，参与的方式有很多，请参考我们的贡献指南。
### 联系我们

flagperf@baai.ac.cn
### 许可证

本项目基于Apache 2.0 license。
本项目部分代码基于MLCommons https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA 实现。
关于各模型测试Case的情况，请参考各模型测试Case目录。
