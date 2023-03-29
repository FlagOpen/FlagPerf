# 昆仑 Bert-Pytorch

- 语言：中文
- 厂商名称：昆仑芯
- 推荐环境配置
  - 硬件
    - 机器型号：R480
    - 芯片/加速卡型号：R300
  - 软件
    - OS类型：Linux
    - OS kernel版本：Linux 5.4.0-26-generic x86_64
    - Docker 版本：20.10.9
    - 加速卡监控命令：xpu_smi
- 厂商对于模型的适配
  - 算子扩展
  - 算子优化：基于MILR的算子融合、图优化
  - 数值精度：float32
  - 支持的分布式训练方式: 单卡、单机多卡

| 分布式训练方式 | 启动命令【含参数】 |
| :-----| :---- | 
| 单机单卡 | python -m torch.distributed.launch --nproc_per_node 1 --master_port=12355 run_pretraining.py --do_train --data_dir=dataset --extern_config_dir=/home/FlagPerf/training/kunlun/bert-pytorch/config --extern_config_file=config_R200x1x1.py --enable_extern_config |
| 单机多卡 | python -m torch.distributed.launch --nproc_per_node 2 --master_port=12355 run_pretraining.py --do_train --data_dir=dataset --extern_config_dir=/home/FlagPerf/training/kunlun/bert-pytorch/config --extern_config_file=config_R200x1x2.py --enable_extern_config |

  - 训练加速方式
  - 训练收敛情况
    - target_acc: 0.720
    - converged_acc: 0.739
    - epoch: 2
    - loss: 1.174
    - seed: 9031
    - Intial LR: 1.4e-4
    - batch-size: 12
