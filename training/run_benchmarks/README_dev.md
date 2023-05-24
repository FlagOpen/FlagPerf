# dev.py功能说明

### 总体概述

1. 不考虑标准输出的内容，dev.py在功能上与run.py完全一致。即：运行dev.py，也能够执行testconfig中的对应case
2. dev.py仅在标准输出上与run.py有区别。dev.py屏蔽了host所有的标准输出（不包括容器内的，因此不影响result中的log），改为输出以下信息：

```shell
Command 1, run at host
	xxx
Command 2, run at host
    xxx
Command 3: run at host
[INFO] Command 3 let you go into docker(container)
    xxx
Command 4, run at docker(container)
[INFO] If you set nnodes != 1, you should run command 1-3 on each hosts, then run the corresponding command 4 respectively
    Command 4 at host yyy:
        xxx
    Command 4 at host zzz:
        xxx
```

### 设计思路

* 区分在容器外和容器内的执行命令，提供容器内执行命令，方便开发者直接在容器中调试。
* 在“case尚未添加完毕”的时候，方便开发者跳过flagperf启动、加载、检查等环节，直接执行自己case对应的task

### 简单使用步骤

##### 验证已有case

0. 检查case路径设置，如CASES = {"cpm:pytorch:A100:1:8:1": "/home/datasets_ckpt/cpm/train/"}, 当字典中多个k-v时，行为与执行run.py一致。会按照python解释器的字典key值遍历顺序遍历所有case，依次输出对应case的四条命令、执行case（建议使用dev.py功能时，在CASES中只放置一对儿k-v）

1. 已有case的文件结构已经设置正确。因此直接在运行这一case的时候，将run.py更改为dev.py，即可获取到dev.py的专有输出，获取相应命令
2. 可以在相应主机上，运行command1-4，手动完成“run.py”的大部分流程
3. 在command4运行完后，可继续执行command4，再次运行这一case，不需要关闭现有容器、启动新容器、配置新容器等步骤

##### 添加新case过程中

* 本文件的主要功能时，在case尚未正确添加的过程中，帮助开发者跳过相关流程

1. 在test_config中写好即将添加的case
2. 在nvidia/下面添加“{model}-{framework}/”目录，在{model}-{framework}/目录下添加config/与extern/目录，在config/目录下添加config_{hardware}x{nnode}x{nproc}.py
3. 在benchmarks/下面添加{model}/目录，在{model}/目录下添加{framework}/目录，在{framework}/目录下添加run_pretraining.py
4. 此时即可执行dev.py，获取在flagperf框架中，启动当前配置的case所需的命令，并进行调试
5. 后续开发可仅在benchmarks/{model}/{framework}/下面进行，无需再关注flagperf框架

* 下面以添加faster_rcnn模型pytorch框架标准case（nvidia A100 1\*8）作为样例。添加前首先将数据集、backbone权重等文件存放在了/home/xxx/目录下

	1. 在test_config中，将CASES写为{'faster_rcnn:pytorch:A100:1:8:1':'/home/xxx'}

 	2. 在nvidia/下面添加faster_rcnn-pytorch/，在faster_rcnn-pytorch/目录下添加config/与extern/目录，在config/目录下添加config_A100x1x8.py
 	3. 在benchmarks/下面添加faster_rcnn/目录，在faster_rcnn/目录下添加pytorch/目录，在pytorch/目录下添加run_pretraining.py
 	4. 运行dev.py 获取了包含4条命令的输出

此时，已经完成了dev.py的功能

5. 输入命令1-3
6. 输入命令4。因为此时run_pretraining.py是空的，因此不会有任何效果
7. 在benchmarks/faster_rcnn/pytorch/下面添加各种trainer/，dataloader/，model/等目录，按照文档标准填写run_pretraining.py，完成case编写
8. 在步骤7的过程中，可反复使用命令4进行调试
9. 调试完毕，退出容器，编写nvidia/faster_rcnn-pytorch/下面各文件
10. 进行完整验证，填写case readme，提交PR