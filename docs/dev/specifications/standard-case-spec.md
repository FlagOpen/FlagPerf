# 标准Case规范

> 文档信息说明
>- 文档面向人群：标准Case的开发人员
>- 文档目的：给出实现标准Case的规范，降低标准case开发成本，提升共建项目的可维护性。

## 1.标准Case定义

标准case是模型-训练框架-英伟达的一个组合，以下简称**case。** 标准case以nvidia GPU作为运行参照，代码实现上面依赖cuda，**原则上**不依赖其它特定芯片厂商的软件包，如NVIDIA/apex相关实现，则不该出现在标准case中。

标准Case代码路径位于training/benchmarks/&lt;model&gt;/&lt;framework&gt;。

对于不同类型的芯片，芯片厂商需要扩展标准Case的模块和接口，完成适配。

## 2.标准Case的模型选择

- FlagPerf旨在和大家共同探索开源、开放、灵活、公正、客观的AI芯片评测体系，建立评测软件生态，提供行业价值，因此在模型选择上面考虑以下几个方面：

- 尽量覆盖典型应用场景典型模型
- 及时跟进新的热门的模型，便于用户及时评测
- 模型代码标准实现源自github公开高认可的仓库

## 3. 标准Case实现规范

### 3.1 代码结构
标准Case实现路径在training/benchmarks/&lt;model&gt;/&lt;framework&gt;/下，仅为NVIDIA上基础版本实现，厂商可以通过扩展模型实现的接口来适配自己的芯片，具体参见[厂商适配Case的规范（试行讨论版）](https://qgsq1vkxhz.feishu.cn/docx/TEYddkMWko2qQExZVgsc09QGnSd) 

标准Case代码以run_pretraining.py脚本为入口，该脚本由start_&lt;framework&gt;_task.py脚本在容器内调用执行。整个代码建议遵循常见的训练pipeline，以glm-pytorch case为例，包括以下几部分：

```Bash
├── glm
│   └── pytorch
│       ├── config   # 【必选】case基本配置目录
│       │   ├── __init__.py
│       │   ├── _base.py  # case基础配置：包含了case运行的所有配置参数
│       │   └── mutable_params.py # 可变参数列表：定义了厂商可以覆盖的配置项名称。
│       ├── dataloaders # dataloaders定义构建dataset, dataloader相关逻辑（train && evaluate）
│       │   ├── __init__.py
│       │   └── dataloader.py  # dataloader: 定义dataset, dataloader的构建逻辑，具体函数名参考具体参考run_pretraining.example.py中说明
│       ├── model    # 模型定义
│       │   ├── __init__.py
│       │   ├── layers
│       │   └── models
│       ├── optimizers # 优化器相关
│       │   ├── __init__.py
│       │   ├── fp16_optimizer.py # fp16优化器
│       │   └── loss_scaler.py    # loss缩放，解决FP16类型数据下溢问题。
│       ├── readme.md  # 【必选】case readme文档
│       ├── run_pretraining.py  # 【必选】执行训练入口脚本，可以使用dev/docs/run_pretaining.example.py作为模版
│       ├── schedulers  # lr schedulers
│       │   ├── __init__.py
│       │   └── base.py
│       └── train
│           ├── __init__.py
│           ├── evaluator.py  #  evaluator: 定义验证evaluate方法
│           ├── trainer.py    # trainer： 定义训练pipeline
│           ├── trainer_adapter.py # trainer_adapter：定义训练流程的各个步骤的具体实现 
│           └── training_state.py  # training_state：保存训练状态的dataclass
```

所有标准Case需要在Nvidia GPU环境验证通过，如实现Nvidia相关的扩展，如apex等NVIDIA额外支持的包，则可参考 [厂商适配Case的规范（试行讨论版）](https://qgsq1vkxhz.feishu.cn/docx/TEYddkMWko2qQExZVgsc09QGnSd)。

### 3.2 添加规范
* 总结流程如下
  1. 下载数据集和checkpoint(如有)
  2. 从零(或checkpoint)开始训练，保存ckpt
  3. 验证精度达标
  4. 从原始仓库分离模型、config
  5. 整理trainer、trainer_adapter、run_pretraining、config
  7. 撰写nvidia-1x8 config
  8. 测试1x8全流程（结果应与1、2相同）
  9. 测试1x1,2x8
  10. 补充case文档，模型文档
  11. 对照PR提交规范，提交PR

以下为详细解释。
#### 0) 准备工作

* 环境准备
  优先使用Perf已有的镜像版本，缺包可基于已有镜像新增安装包。


* 原始代码训练验证

  确定待添加的模型代码链接，使用**原始代码**的数据集和代码配置进行**单机8卡**模型复现。
  
  使用原始代码的原因：
  * 确保源代码没有质量问题避免无用功 
  * 确定目标精度，若NV的精度在合理值，则该值作为其他厂商对标精度，且作为配置文件中target_acc的值

* 数据集记录

  * 将数据集下载路径写入文档，便于复现结果。若原始代码不带数据集，使用业内公开知名数据集。

* 【重要】checkpoint保存
    
  * 每**10**个保存一次训练ckeckpoint用于Perf的finetune,提交代码时候将选定ckpt 交给智源方Reviewers.
  * 最终提交的模型case期望在不同厂商芯片上都可以在【2-5h】内收敛精度达标，因此建议推算使用合适的checkpoint resume训练。

**验证原始代码没有问题和收集到目标ckpt之后，开始Perf的适配工作。**

注: 
- checkpoint文件在PR提交时候交给智源方上传到公开地址供用户下载，ckpt文件命名方式：模型-框架-md5值.pth


#### 1) 实现模型训练主体逻辑


在training/benchmarks下添加&lt;model&gt;/&lt;framework&gt;子目录，pytroch和paddle的标准case可参考下面的目录结构组织代码：

```Bash
.
├── config      # 【必选】case基本配置目录
├── dataloaders
├── model
├── optimizers
├── readme.md   # 【必选】case文档，规范参考 https://qgsq1vkxhz.feishu.cn/docx/NMAGdJ3w6oltKJxJB9ccVL43n5c
├── run_pretraining.py #【必选】执行训练入口脚本，可以使用dev/docs/run_pretaining.py.example作为模版
├── schedulers
└── train
```

在training/nvidia/&lt;模型&gt;-&lt;框架&gt; 目录下
```Bash
.
├── config # nv下的配置文件
└── extern # nv上的特有实现，如apex优化
```

其它框架如tensorflow2, 以resnet50为例:

```Bash
#以/FlagPerf/training/benchmarks/resnet50/tensorflow2为例：
├── config      # 【必选】tf2-example下的config相关基类
├── core        # 训练pipeline的核心工具组件，以trainer_adapter.py为代表
├── modeling    # 模型pipeline各组成模块参考实现：optimization、activations
├── resnet.     # resnet 实现代码，该处保持和tf官方一致。
├── readme.md   # 【必选】case文档，规范参考 https://qgsq1vkxhz.feishu.cn/docx/NMAGdJ3w6oltKJxJB9ccVL43n5c
├── run_pretraining.py #【必选】执行训练入口脚本，可以使用dev/docs/run_pretaining.py.example作为模版
└── utils   #公共工具包
```


#### 2) 实现关键代码逻辑

复制training/benchmarks/&lt;model&gt;/&lt;framework&gt;/run_pretraining.py.example 为training/benchmarks/&lt;model&gt;/&lt;framework&gt;/run_pretraining.py脚本.

根据该脚本中的标记TODO的位置进行修改，串接整个训练pipeline，保证关键接口不变的情况下，自定义内部实现。

这部分链接到文件模块trainer、trainer_adapter、run_pretraining、config, 都属于benchmark 必须项。

该config模块为benchmark case 1*8训练配置，且是和硬件厂商无关的配置，凡和硬件厂商有关的配置，放置于厂商config目录下。

提交的最终版本代码，需要由ckpt开始训练，保证2-5内在NV上训练达标【重要】。

#### 3) 添加NVIDIA的配置

- 文件路径：training/nvidia/&lt;model&gt;-&lt;framework&gt;/config
- 文件内容：不同配置下的模型超参数及硬件依赖参数
- 配置文件列表如下：（以GLM-Pytorch为例）

```Bash
├── config_A100x1x1.py      # 必选, 单机单卡配置，性能结果和精度验证
├── config_A100x1x8.py      # 必选, 单机8卡配置，性能结果和精度验证
├── config_A100x2x8.py      # 必选, 2机8卡配置，性能结果和精度验证
├── environment_variables.sh # 环境变量文件，在容器启动后，运行case之前会source该文件，设置容器中的环境变量。
└── requirements.txt         # 容器中安装python依赖包。FlagPerf框架在启动容器后，在运行case之前，执行pip install -r requirements.txt，安装依赖库
```

以上工作完成，满足进入容器中启动训练任务的目标，下面的工作保证在容器外能以统一方式批量启动测例。

#### 4) 添加测例入口配置

1. 在training/run_benchmarks/config/test_conf.py中添加新标准case的key-value, 验证以run.py 方式启动有效性。

以GLM和Bert为例,

```Bash
# Set the case dict you want to run here.
'''
# Users must use {
    "model:framework:hardwareID:nnodes:nproc:repe": "dataset path"}
'''
CASES = {
    "bert:pytorch:A100:1:8:1": "/home/datasets_ckpt/bert/train/",
    "glm:pytorch:A100:1:8:1": "/home/datasets_ckpt/glm/train/",
}
```

2. 在training/run_benchmarks/config/cluster_conf.py中添加标准case运行机器

```Python
# Hosts to run the benchmark. Each item is an IP address or a hostname.
HOSTS = ["10.209.20.12","10.209.20.13"]
```

3. 【必需】在run.py 脚本启动完整测试

```Bash
python3 ./run_benchmarks/run.py
```

注：调试运行时可在容器中进行，最终提交前需run.py 完整测试，验证代码工作正常。关于配置方法，可参考：https://github.com/FlagOpen/FlagPerf#readme

#### 4) 验证达标要求

- 以Perf方式训练模型收敛达到目标精度(NV上原始代码单机8卡精度值)
- 单个case的收敛时间在2-5h内
- 多机/多卡吞吐量加速比符合预期
- 有训练过程和benchmark结果日志输出（nv机器上），包括训练中的N个steps的和最终结果输出。finished_info包括不限于：e2e_time、training_sequences_per_second、 converged、final_accuracy、raw_train_time、init_time
- 有可用的NVidia GPU适用的配置样例

### 3.3 配置文件规范

涉及标准Case实现的配置有2个：
* 基本配置: 基本配置是模型训练及运行环境相关的配置参数，主要分为两大类:模型超参(lr等)和训练配置(log_freq等)，路径为training/benchmarks/&lt;模型&gt;/&lt;框架&gt;/config/_base.py和mutable_params.py, 其中mutable_params.py 中定义的参数表示可覆盖参数项

* Nvidia适配的配置: Nvidia适配的配置是指标准Case运行在Nvidia GPU环境的配置参数。作为Nvidia的标准配置，可以和基础配置一致，也可以在运行时覆盖标准Case的基础配置参数。

由于FlagPerf的一些代码设定，对配置文件路径和内容有一定要求。

#### 1）`case基本配置`

- `标准Case基本配置`
  - 路径：&lt;model&gt;-&lt;framework&gt;/config/_base.py 。**定义模型训练相关的所有参数，及case运行环境需要的基本参数**。模型名称、模型结构、数据集、checkpoint、超参数、分布式参数等。
  -  配置项说明如下，可参照[standard-case-config-base.py.example](../standard-case-config-base.py.example)：
    - 必选参数：
      - `vendor`，值为"nvidia"即可，会在运行FlagPerf时被配置在test_conf里的vendor值覆盖。
      - `data_dir`，值为""空即可，会在运行FlagPerf时被配置在test_conf中对应case配置项data_dir_container覆盖。
      - `init_checkpoint`：初始化模型checkpoint文件，填写相对`data_dir`的路径，一般提供文件名即可，下载checkpoint文件后放置于`data_dir`
    - 可选参数：
      - `train_data`：训练数据路径，填写相对`data_dir`的路径
      - `eval_data`：评估数据路径，填写相对`data_dir`的路径

- `可改写配置项`
  - 路径：&lt;model&gt;-&lt;framework&gt;/config/mutable_params.py。**定义厂商（含nvidia）可覆盖的_base中参数列表。**主要是和vendor和运行环境相关的配置项，定义为mutable_params数组。
    - 厂商可以在training/&lt;vendor&gt;/&lt;model&gt;-&lt;framework&gt;/config/config_xxxx.py中，重新定义参数值，从而实现对于case的基本配置_base.py配置参数的**覆盖**。
    - 例如：mutable_params = ['vendor', 'local_rank', 'train_batch_size']，其中**vendor**为必选项。

#### 2）`Nvidia适配的配置`

在Nvidia GPU上运行所需的配置文件放在training/nvidia/&lt;model&gt;-&lt;framework&gt;/config目录下，可以看作是Nvidia适配标准Case的配置项，由于训练规模不同，可以给出多个配置文件。在FlagPerf运行时，会根据test_conf里的case配置项选择加载哪个配置文件。

配置文件命名为：config_&lt;machine_model&gt;x&lt;nnodes&gt;x&lt;nproc&gt;.py，例如单机4卡的A100环境运行，使用config_A100x1x4.py，这里主要放置是厂商适配case时可覆盖的参数，一般定义在自己设备上跑该Case最优的配置。

此外，如果该标准Case在预先构建的nvidia镜像中无法直接运行，需要一定的环境配置和依赖包安装，请添加environment_variables.sh和requirements.txt。具体可以参考：[厂商适配Case的规范](case-adatpion-spec.md) 。

### 3.4 文档要求

模型README文档（首次添加该模型的case时，需要填写）及 case README文档 符合文档模版要求。文档模版请参考： [模型README文件模版](../readme-templates/model-readme-template.md) 和 [case README文件模版 ](../readme-templates/case-readme-template.md)。



## 4. 代码提交与Review合并

FlagPerf采用开源共建的方式，开发者应fork [FlagPerf仓库](https://github.com/FlagOpen/FlagPerf/tree/main) 到个人帐号下，修改代码&验证通过后，提交PR给FlagPerf项目指给reviewers。FlagOpen相关研发人员Review通过后，合并代码。具体操作可参照：https://docs.github.com/en/get-started/quickstart/contributing-to-projects

### 4.1 标准case提交内容检查
  #### 首次添加case
  1. 只提交添加模型必要代码变动，代码格式执行"yapf -i --style "pep8" --recursive ./FlagPerf "
  2. 文档齐全，包括模型、case(包括1x1, 1x8, 2x8 性能精度结果)、厂商(如需)三个
  3. 提交ckpt到智源方(由智源方上传公开网站供下载)
  4. 提供到1x1，1x8，2*8精度log给智源方用于存档

  #### 修改标准case
  1. 如果Perf中已经存在标准实现, 需要改动标准实现且修改内容影响Case运行结果，请在Case的README.md更新新的运行记录，随PR提交;并建议在PR的comment里提交在Nvidia GPU上运行日志附件。
  2. 如果该case已经有厂商已经适配，需要评估该修改对所有已经适配的厂商扩展是否有影响。

### 4.2 标准case的PR提交规范
  1. PR提交请说明PR的作用/目的
  2. 如果修改内容影响Case NV运行结果，请在标准Case的README.md更新的运行记录，随PR提交GPU上运行日志附件。
  3. 如果修改内容预判可能影响其他厂商适配，请在PR里注明或联系Reviewer。
