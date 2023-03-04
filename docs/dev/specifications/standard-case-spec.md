# 标准Case规范

> 文档信息说明

> - 文档面向人群：标准Case的开发人员

> - 文档目的：给出实现标准Case的规范，降低标准case开发成本，提升共建项目的可维护性。

## 1. 什么是标准Case

标准case是模型-训练框架的一个组合，以下简称**case。** 标准case以nvidia GPU作为运行参照，代码实现上面依赖cuda，但原则上不依赖其它特定芯片厂商的软件包。

标准Case代码路径位于training/benchmarks/<model>/<framework>。

对于不同类型的芯片，需要芯片厂商需要扩展标准Case的模块和接口，完成适配。

## 2. 标准Case的合作共建方式

### 	2.1 标准Case的选择

- FlagPerf旨在和大家共同探索开源、开放、灵活、公正、客观的AI芯片评测体系，建立评测软件生态，提供行业价值，因此在Case选择上面考虑以下几个方面：

- - 需尽是覆盖典型应用场景，且包含典型场景的常用模型
  - 及时跟进新的热门的模型，便于用户测试

###     2.2 合作共建机制

考虑到可选模型众多，且共建团队众多 ，为了避免大家开发内容冲突，智源会定期与大家讨论模型列表并确认大家的分工。后续项目更成熟可能考虑在社区发布Issue大家标记认领。

代码提交和合并直接在github上进行，具体可参照后面的代码提交与Review合并流程。

###     2.3 一般性的原则

FlagPerf项目还在起步阶段，各种标准和规范还不健全，项目代码也在持续重构和优化中，有任何建议、意见或疑问请随时提出讨论。

标准Case原则上定义了面向芯片的benchmark的workload，且需要与不同芯片厂商适配，因此不能绑定特定芯片实现。

## 3. 标准Case实现规范

### 3.1 代码规范

#### 1) 代码和配置文件目录结构

标准Case实现路径在training/benchmarks/<model>/<framework>/下，厂商可以通过扩展模型实现的接口来适配自己的芯片，具体参见[厂商适配Case的规范（试行讨论版）](https://qgsq1vkxhz.feishu.cn/docx/TEYddkMWko2qQExZVgsc09QGnSd) 

标准Case代码以run_pretraining.py脚本为入口，该脚本由start_<framework>_task.py脚本在容器内调用执行。整个代码建议遵循常见的训练pipeline，以glm-pytorch case为例，包括以下几部分：

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

为了开发验证，同时也为了给用户提供运行参考，所有标准Case需要带Nvidia GPU的扩展适配，并在Nvidia GPU环境验证通过。如需实现Nvidia相关的扩展，可参考 [厂商适配Case的规范（试行讨论版）](https://qgsq1vkxhz.feishu.cn/docx/TEYddkMWko2qQExZVgsc09QGnSd)。

#### 2) 单元测试

暂不做要求。

#### 3) 代码风格

1. Python代码使用python pep8风格，可以使用yapf进行格式化，例如：

```Bash
yapf -i --style "pep8" --recursive ./FlagPerf
```

### 3.2 添加方法

#### 1) 实现模型训练主体逻辑

在training/benchmarks下添加<model>/<framework>子目录，pytroch和paddle的标准case可参考下面的目录结构组织代码：

```Bash
.
├── config      # 【必选】case基本配置目录
├── dataloaders 
├── model
├── optimizers
├── readme.md   # 【必选】case文档，规范参考 https://qgsq1vkxhz.feishu.cn/docx/NMAGdJ3w6oltKJxJB9ccVL43n5c
├── run_pretraining.py #【必选】执行训练入口脚本，可以使用dev/docs/run_pretaining.example.py作为模版
├── schedulers
└── train
```

其它框架如tensorflow2等的参考目录结构待补充。

#### 2) 实现训练入口程序

复制[docs](https://github.com/yuzhou03/FlagPerf/tree/pretrain-example-helper/docs)/[dev](https://github.com/yuzhou03/FlagPerf/tree/pretrain-example-helper/docs/dev)/run_pretraining.example.py为training/benchmarks/<model>/<framework>/run_pretraining.py脚本，根据该脚本中的标记TODO的位置进行修改，串接整个训练pipeline。

#### 3) 添加NVIDIA的配置

- 文件路径：training/nvidia/<model>-<framework>/config
- 配置文件列表如下：（以GLM-Pytorch为例）

```Bash
├── config_A100x1x2.py      # 单机2卡配置
├── config_A100x1x8.py      # 单机8卡配置
├── config_A100x2x8.example # 2机8卡配置示例
├── config_A100x2x8.py      # 2机8卡配置
├── environment_variables.sh # 环境变量文件，在容器启动后，运行case之前会source该文件，设置容器中的环境变量。
└── requirements.txt         # 容器中安装python依赖包。FlagPerf框架在启动容器后，在运行case之前，执行pip install -r requirements.txt，安装依赖库
```

#### 4) 测试验证

在training/run_benchmarks/config/中添加测试Case相关配置，并通过training/run_benchmarks/run.py脚本启动测试。验证代码工作正常，并符合测试达标要求即可提交PR。关于配置方法，可参考：https://github.com/FlagOpen/FlagPerf#readme

### 3.3 数据集的选择

 数据集通常选用：

- 论文中的使用的数据集
- 知名公开实现使用的数据集

### 3.4 模型Checkpoint的要求

- 如果有该case的框架对应的checkpoint文件，需提供下载链接，最好提供原始checkpoint文件的md5值，以便校验下载后文件的完整性
- 如果只有其他框架的checkpoint文件，例如添加的是bert-pytorch的case，只有tf2的checkpoint可供下载。需在README文档里提供tf2的**checkpoint下载地址、转换工具/脚本，**以及**转换的命令。**

### 3.5 配置文件规范

涉及标准Case实现的配置有2个：基本配置 和 Nvidia适配的配置。基本配置是模型训练及运行环境相关的配置参数，Nvidia适配的配置是指标准Case运行在Nvidia GPU环境的配置参数。后者在运行时会覆盖标准Case的基础配置参数。

由于FlagPerf的一些代码设定，对配置文件路径和内容有一定要求。

#### 1）case基本配置

- 标准Case基本配置
  - 路径：<model>-<framework>/config/_base.py 。**定义模型训练相关的所有参数，及case运行环境需要的基本参数**。模型名称、模型结构、数据集、checkpoint、超参数、分布式参数等。
  -  配置项说明如下，可参照docs/dev/standard-case-config-base.py.example：
    - 必选参数：
      - **vendor，**值为"nvidia"即可，会在运行FlagPerf时被配置在test_conf里的vendor值覆盖。
      - **data_dir，**值为"/home/datasets"即可，会在运行FlagPerf时被配置在test_conf中对应case配置项data_dir_container覆盖。
    - 可选参数：
      - **train_data**：训练数据路径，填写相对**data_dir**的路径
      - **eval_data**：评估数据路径，填写相对**data_dir**的路径
      - **init_checkpoint**：初始化模型checkpoint，填写相对**data_dir**的路径
      - 其它模型训练相关参数，例如初始learning rate等。
- 可改写配置项
  - 路径：<model>-<framework>/config/mutable_params.py。**定义厂商（含nvidia）可覆盖的_base中参数列表。**主要是和vendor和运行环境相关的配置项，定义为mutable_params数组。
    - 厂商可以在training/<vendor>/<model>-<framework>/config/config_xxxx.py中，重新定义参数值，从而实现对于case的基本配置_base.py配置参数的**覆盖**。
    - 例如：mutable_params = ['vendor', 'local_rank', 'train_batch_size']，其中**vendor**为必选项。

#### 2）Nvidia适配的配置

在Nvidia GPU上运行所需的配置文件放在training/nvidia/<model>-<framework>/config目录下，可以看作是Nvidia适配标准Case的配置项，由于训练规模和训练方法不同，可以给出多个配置文件。在FlagPerf运行时，会根据test_conf里的case配置项选择加载哪个配置文件。

配置文件命名为：config_<machine_model>x<nnodes>x<nproc>.py，例如单机4卡的A100环境运行，使用config_A100x1x4.py，这里主要放置是厂商适配case时可覆盖的参数，一般定义在自己设备上跑该Case最优的配置。

此外，如果该标准Case在预先构建的nvidia镜像中无法直接运行，需要一定的环境配置和依赖包安装，请添加environment_variables.sh和requirements.txt。具体可以参考：[厂商适配Case的规范（试行讨论版）](https://qgsq1vkxhz.feishu.cn/docx/TEYddkMWko2qQExZVgsc09QGnSd) 。

### 3.6 测试达标要求

- 可使用FlagPerf正常配置并运行训练
- 训练可收敛到目标精度
- 有训练过程和benchmark结果日志输出（nv机器上），包括训练中的N个steps的和最终结果输出
- 有可用的NVidia GPU适用的配置样例

### 3.7 文档要求

模型README文档（首次添加该模型的case时，需要填写）及 case README文档 符合文档模版要求。文档模版请参考： [模型README文件模版](https://qgsq1vkxhz.feishu.cn/docx/GDVBdQVPmo4RcSxT4sjcVwDunvg) 和 [case README文件模版 ](https://qgsq1vkxhz.feishu.cn/docx/NMAGdJ3w6oltKJxJB9ccVL43n5c)。

## 4. 代码提交与Review合并

FlagPerf采用开源共建的方式，开发者应fork [FlagPerf仓库](https://github.com/FlagOpen/FlagPerf/tree/main) 到个人帐号下，修改代码&验证通过后，提交PR给FlagPerf项目。FlagOpen相关研发人员Review通过后，合并代码。具体操作可参照：https://docs.github.com/en/get-started/quickstart/contributing-to-projects