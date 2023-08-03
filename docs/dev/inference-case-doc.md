# 文档目录结构即说明

> 1. 标准case定义
> 2. 标准case执行流程（用户手册）
> 3. 标准case开发规范（开发手册）
> 4. 适配case流程（厂商适配文档）
>
> 下述表达中，所有“原则上”的内容，在实际执行中不可修改。如想修改，请联系项目所有者进行商讨，确定修改后本文档将进行对应更新。即“原则上”部分可进行框架本身的优化

## 1.标准Case定义

case是”模型-训练框架验证-编译器推理“的一个组合，以下简称**case。** 标准case以（pytorch+nvidia A100）为训练框架验证部分，以nvidia tensorrt为编译器推理部分。训练框架验证代码实现上面依赖cuda，**原则上**不依赖其它特定芯片厂商的软件包，如NVIDIA/apex相关实现，不该出现在标准case中。编译器推理代码实现上除编译器类自身外，原则上不依赖包括cuda在内的任意GPU、nvidia相关软件。

标准Case代码路径位于inference/benchmarks/&lt;model&gt;/&lt;framework&gt;。

对于不同类型的芯片，芯片厂商需要通过配置文件跳过训练框架验证及有关部分，利用一次性开发的编译器类完成推理。

## 2. 标准Case执行流程

### 2.1 代码结构
FlagPerf推理框架代码包括**标准case代码**（benchmarks）、配置（configs）、容器配置（docker_images）、编译器及编译运行时（inference_engine）、容器内工具（tools）、容器外工具（utils），以及容器内外两个进程入口：run.py(容器外)、run_inference.py(容器内)。此外，还有onnxs/、result/等用于存放临时文件的目录。代码结构如下所示：

```Bash
.
├── benchmarks   # 标准case代码
│   └── resnet50
│       ├── README.md
│       └── pytorch
│           ├── __init__.py
│           ├── __pycache__
│           ├── dataloader.py
│           ├── evaluator.py
│           ├── export.py
│           ├── forward.py
│           └── model.py
├── configs   # 配置
│   ├── host.yaml
│   └── resnet50
│       ├── configurations.yaml
│       ├── parameters.yaml
│       └── vendor_config
│           └── nvidia_configurations.yaml
├── docker_images   # 容器配置
│   └── nvidia
│       ├── __pycache__
│       ├── nvidia_analysis.py
│       ├── nvidia_monitor.py
│       ├── pytorch_1.13
│       │   ├── Dockerfile
│       │   └── pytorch1.13_install.sh
│       └── pytorch_2.1
│           ├── Dockerfile
│           └── pytorch2.1_install.sh
├── inference_engine   # 编译器及编译运行时
│   └── nvidia
│       ├── __pycache__
│       ├── tensorrt.py
│       └── torchtrt.py
├── onnxs   # 临时目录，存放中间结果
├── result   # 临时目录，存放log及评测结果
├── run.py   # 容器外入口
├── run_inference.py   # 容器内入口
├── tools   # 容器内工具
│   ├── __init__.py
│   ├── __pycache__
│   ├── config_manager.py
│   ├── init_logger.py
│   └── torch_sync.py
└── utils   # 容器外工具
    ├── __init__.py
    ├── __pycache__
    ├── cluster_manager.py
    ├── container_manager.py
    ├── image_manager.py
    ├── prepare_in_container.py
    ├── run_cmd.py
    └── sys_monitor.py

```

### 2.2 执行流程

##### 2.2.1 配置config文件

1. 在任何时刻，FlagPerf推理框架中仅有configs/目录下的yaml文件可被用户配置

2. 用户需要根据自身环境，配置configs/host.yaml:

```Bash
FLAGPERF_PATH: "/home//FlagPerf/inference" # FlagPerf/inference所处绝对路径
FLAGPERF_LOG_PATH: "result"  # 存放log及评测结果的目录，需要提前mkdir建好
VENDOR: "nvidia"  # 待测硬件。标准case选择“nvidia”
FLAGPERF_LOG_LEVEL: "INFO"   # Log级别，推荐选择“INFO”或“DEBUG”
LOG_CALL_INFORMATION: True   # Log时是否显示caller信息，如某函数某行，推荐选择True
HOSTS: ["10.1.2.155"]    # 待测硬件所属ip
SSH_PORT: "22"   # 待测硬件ssh端口，推荐不进行修改
HOSTS_PORTS: ["2222"]    # 待测硬件主机端口，推荐不进行修改
MASTER_PORT: "29501"    # 待测硬件服务器pytorch master端口，推荐不进行修改
SHM_SIZE: "32G"    # 启动容器共享内存，推荐不进行修改
ACCE_CONTAINER_OPT: " --gpus all"    # 计算卡选择。标准case选择“--gpus all”
PIP_SOURCE: "https://mirror.baidu.com/pypi/simple"   # 容器内pip源，服务器在中国大陆地区则推荐不进行修改
CLEAR_CACHES: True   # 评测选项，推荐用户不进行修改，开发者可酌情选择False
ACCE_VISIBLE_DEVICE_ENV_NAME: "CUDA_VISIBLE_DEVICES"    # 计算卡可见性环境变量key。标准case选择“CUDA_VISIBLE_DEVICES”

# CASES为一个dict。每个key确定了运行case的case名称及训练框架版本，value为待挂载进容器的目录，原则上为数据集、模型权重的公共父目录。key的格式为"<case>:<framework>"
CASES:
    "resnet50:pytorch_1.13": "/raid/dataset/ImageNet_1k_2012/val"
```


3. 在进行上述配置时，VENDOR、\<case\>、\<framework\>确定了一组评测对象，在进行配置时，需要首先确保：

   a. docker_images下有VENDOR目录

   b. inference_engine下有VENDOR目录

   c. benchmarks、configs下有\<case\>目录

   d. docker_images/VENDOR下有\<framework\>目录

   上述四者均满足时，说明评测对象已经在FlagPerf中开发完毕，可以进行评测

4. 用户需要根据评测对象，配置configs/\<case\>/configuration.yaml

```Bash
batch_size: 256
# 3*224*224(1 item in x)
input_size: 150528
fp16: true
compiler: tensorrt
num_workers: 8
log_freq: 30
repeat: 5
# skip validation(will also skip create_model, export onnx). Assert exist_onnx_path != null
no_validation: false
# set a real onnx_path to use exist, or set it to anything but null to avoid export onnx manually(like torch-tensorrt)
exist_onnx_path: null
# set a exist path of engine file like resnet50.trt/resnet50.plan/resnet50.engine
exist_compiler_path: null
```


5. 在进行上述配置时，需要确保inference_engine/VENDOR下有"compiler".py文件。
6. 在评测标准case时，原则上保持no_validation=false, exist_onnx_path=null, exist_compiler_path=null。在评测适配case时，通常的配置为no_validation=true, exist_onnx_path=\<path_to_your_onnx_file\>, exist_compiler_path=null。对于非开发者来说，**原则上不推荐更改非host.yaml的任何配置选项**。

##### 2.2.2 容器外执行流程

1. 用户在完成配置后，仅需在FlagPerf/inference目录下执行sudo python3 run.py即可启动对应评测。
2. run.py会在执行一系列环境准备后，根据host.yaml启动一个合适的容器、在容器内启动1个run_inference.py对应进程，并根据硬件启动一个或若干个监视器进程。
3. 在容器内进程执行结束后，run.py会将其结果与监视器进程结果进行汇总，形成评测结果进行输出。

##### 2.2.3 容器内执行流程

1. 容器内执行流程分为7步，均在run_inference.py中实现。依次为：

   a. 初始化logger、config等

   b. 创建数据集与模型

   c. 训练框架验证

   d. 模型导出为onnx

   e. 编译器编译onnx

   f. 编译器运行时推理

   g. 汇总结果，记录

2. 能够影响流程本身执行情况的配置选项共有4个：compiler、no_validation、exist_onnx_path、exist_compiler_path，均位于configs/\<case\>/configuration.yaml。当评测标准case，即4个配置项依次为tensorrt、false、null、null时，按照上述7步骤依次执行。否则：

   a. compiler可设置为null。执行流程会在完成“c. 训练框架验证”后直接进入"g. 汇总结果，记录"

   b. no_validation可设置为true。此时"b. 创建数据集与模型"中的“模型”将为None，并断言exist_onnx_path不为null，跳过“c. 训练框架验证“与”d. 模型导出为onnx”，直接使用exist_onnx_path给定的onnx文件继续执行“e. 编译器编译onnx”

   c. exist_onnx_path可设置为某具体路径，通常结合no_validation=True使用。在评测非标准case时，因标准case的“c. 训练框架验证“依赖cuda实现，因此可跳过相关步骤

   d. exist_compiler_path可设置为某具体路径，通常结合no_validation=True与exist_onnx_path=“foo”使用。在评测非标准case时，若厂商不希望以统一的onnx格式，而是其编译运行时可理解的具体格式作为直接输入，则可进行如此设置。如no_validation=True、exist_onnx_path=“foo”、exist_compiler_path=\<path_to_your_compiler_engine_file\>，则将在完成"b. 创建数据集与模型"中的数据集部分后，直接跳到“f. 编译器运行时推理”完成推理。
   
3. 下面从4个选项自身，而非整体流程的角度解释4个选项的作用：

   a. compiler为null时，在执行完c步骤后直接跳到g步骤

   b. no_validation为True时，将b步骤中的模型返回为None，跳过c步骤（因断言的存在，实际执行时exist_onnx_path必定不为null，因此必定同时跳过d步骤，防止试图将一个为None的model导出为onnx引发报错）

   c. exist_onnx_path不为null时，跳过d步骤

   d. exist_compiler_path不为null时，跳过e步骤

## 3. 标准Case开发规范

### 3.1 标准case需要开发的内容

除开发框架本身第一个case外，开发其他标准case需要完成的工作包括：

1. 在configs/\<case\>下组织相应配置文件
2. 在benchmarks/&lt;case&gt;/&lt;framework&gt;实现评测流程所需的相应函数
3. 修改host.yaml，完成标准case评测，记录结果，完成benchmarks/&lt;case\>/README.md

### 3.2 各部分开发细则

##### 3.2.0 额外安装包

对于某些case，如需要镜像本身以外的pip包，可在benchmarks/&lt;case\>/\<framework\>下添加requirements.txt，框架启动时会自动在运行此次评测的镜像中安装

##### 3.2.1 config

对于标准case、需要组织的是configurations.yaml、parameters.yaml，及vendor_config/nvidia_configurations.yaml。

1. configurations.yaml

   此文件的配置项原则上应保持与所有其他case（以resnet50）为例严格相同

   batch_size可根据case情况进行调整

   input_size为每个输入包含的元素数量。例如resnet50case中，1个输入为1张3\*224\*224的图片，因此input_size为150528

   fp16原则上为true，不支持fp16的模型或因过多layernorm等算子导致fp16精度下降严重的可选fp16=false

   num_workers通常为4或8

   log_freq原则上保证实际执行过程中，在此log_freq情况下每0.5-1分钟有1次输出即可

   repeat原则上保证实际执行过程中，总时间大于10分钟，小于24小时即可

   对于标准case，compiler、no_validation、exist_onnx_path、exist_compiler_path依次为tensorrt、false、null、null

2. parameters.yaml

   此文件放置大量使用因此需要参数化，但不包含在上述configurations.yaml中的项，每个case不同。例如BERT中的max_seq_length, stable diffusion中的guidance_scale，原则上与硬件无关，各厂商适配case均不可修改

3. vendor_config/nvidia_configurations.yaml此配置项为实现该case时可进行的覆盖性配置，具有如下几条原则：

   a. 项名不可与host.yaml，当前case对应的parameters.yaml存在任何重复

   b. 如项名不在configurations.yaml中（即新添了一项配置，例如nvidia的tensorrt_tmp_path），FlagPerf会报warning来提示

   c. 如项名在configurations.yaml中，且待覆盖的value与原始value不同（即改写了一项配置），FlagPerf会用info进行记录

   d. 如项名在configurations.yaml中，且待覆盖的value与原始value相同（即没有任何实际效果），为保证开发规范，FlagPerf会报error，并直接退出程序。即不允许此种无意义的覆盖。

##### 3.2.2 benchmarks/

对于标准case，需要实现build_dataloader、create_model、evaluator、model_forward、export_model、engine_forward这6个方法，并在benchmarks/\<case\>/\<framework\>/__init\_\_.py中进行import

1. build_dataloader

   根据传入参数config，构造该标准case的dataloader。通常可继承或直接使用torch.utils.data.DataLoader实现

2. build_dataloader

   根据传入参数config，构造该标准case的model。通常为torch.nn.Module及其子类(如transformers.BertForMaskedLM)

3. evaluator

   为一个函数方法。后续通常会将输出pred与真值y传入，计算评测指标。具体传入参数可根据不同case实际情况而定，与本case的model_forward、engine_forward适配。evaluator原则上全过程在CPU上执行，不依赖cuda等。

4. model_forward

   根据传入的model、dataloader、evaluator、config，将dataloader在model上进行前向计算，并使用evaluator获取可量化的计算结果(如bbox mAP、top1 acc)。同时进行总推理时间、纯计算时间的统计，统计方式原则参考本文档3.2.3，实现方式参考resnet50

5. export_model

   根据传入的config，将传入的model导出为onnx文件

6. engine_forward

   入参与实现方式与model_forward一致，只需将其中输出信息log改写为“inference”。输出部分可能需要额外取item[0]。具体细节请参考本文档4.2.1

##### 3.2.3 README

对于标准case，上述开发步骤结束后，需至少完成1条评测记录，并完成README撰写。结构参考benchmarks/resnet50/README.md。其中最重要的是4个性能指标：validation_whole, validation_core, inference_whole, inference_core

1. 两个validation指标为训练框架验证部分的吞吐量，两个inference为编译期运行时推理部分的吞吐量。如设置no_validation，两个validation指标为None。如设置compiler=null，两个inference指标为None
2. 两个whole指标包含了加载数据、计算、评估的部分，两个core仅包含计算部分。通常有价值的指标是inference_core。

## 4. 适配case流程

“适配”的定义为将某厂商硬件及某编译器+运行时纳入FlagPerf推理的评测范围中。厂商硬件第一次适配需要开发容器及监视器，厂商编译器+运行时需要开发编译器类。非首次适配原则上仅需书写配置完成评测记录结果，对于特殊的case可能需要改动编译器类、或开发相应适配器。

### 4.1 第一次适配的额外工作

##### 4.1.1 容器及监视器

厂商某硬件第一次适配时，需要开发容器镜像、计算卡监视器、监视器结果分析三个部分。

1. 容器镜像

   容器镜像以Dockerfile的形式给出。对于推理硬件支持pytorch、cuda等的厂商，可在docker_images/VENDOR/\<framework\>（如docker_images/nvidia/pytorch_1.13）目录下完成DockerFile文件。对于不支持的厂商，可在docker_images/VENDOR/pytorch_foo目录下完成DockerFile文件，并在配置中开启no_validation等选项

2. 计算卡监视器

   计算卡监视器docker_images/VENDOR/VENDOR_monitor.py为一个后台子进程类。可参考docker_images/nvidia/nvidia_monitor.py实现。原则上要记录显存、使用率等信息

3. 监视器结果分析

   上述监视器类在运行时的输出会重定向到文件中，监视器结果分析函数（docker_images/VENDOR/VENDOR_analysis.py 中的analysis_log）函数将接受且仅接受这一重定向文件路径作为参数，需要依次返回运行时最大占用存储、计算卡总存储两个指标

##### 4.1.2 编译器类

厂商某编译器+运行时第一次适配时，需要开发编译器类，位于inference_engine/VENDOR/COMPILER.py，如infereence_engine/nvidia/tensorrt.py。该文件必须实现InferModel类，至少具有\_\_init\_\_，\_\_call\_\_两个方法

1. \_\_init\_\_

   该方法仅会在2.2.3节“e. 编译器编译onnx”步骤被调用1次。推理框架会将config、onnx_path、model（可能为None，如设置了no_validation）传入供编译器进行编译。通常该方法还需要完成运行时上下文的构建。该方法时间不会包含在whole/core吞吐量计算中

2. \_\_call\_\_

   该方法会在2.2.3节“f. 编译器运行时推理”步骤，每个batch被调用1次。每次调用将传入一个list，为该case各个输入。每个输入均为CPU上的torch.Tensor。该方法需要返回这组输入对应的推理结果，返回一个list，为该case的各个输出。每个输出均为CPU上的torch.Tensor。

   某些编译器（如torchtrt）无法使用onnx，而是需要一组输入来确定输入尺寸，因此只能在\_\_call\_\_环节进行编译。因此，编译器类\_\_call\_\_方法的实际输出为result,foo_time。result为各个输出组成的list，foo_time为此阶段与推理无关的时间。原则上foo_time需为0.0。如某编译器的foo_time不为0.0，则此部分会重点review。

### 4.2 正常适配工作

##### 4.2.1 配置覆写

1. 按照标准case开发中nvidia的形式，组织configs/\<case\>/vendor_config/VENDOR_configurations.yaml进行覆写。例如覆写no_validation=true。原则上适配工作不允许直接更改configs/\<case\>/configurations.yaml与parameters.yaml
2. 修改host.yaml，运行评测

##### 4.2.2 记录评测结果，扩展README

将评测结果补充在标准case的README表格的下方，并补充硬件、编译器+运行时参数

提交PR时需附带相应运行log，提交除host.yaml外的文件修改

##### 4.2.3 可能需要的adapter

如编译器类无法通用，则需要开发对应case的适配器，继承原始编译器类进行改写。该功能目前尚无实际案例，等待开发中