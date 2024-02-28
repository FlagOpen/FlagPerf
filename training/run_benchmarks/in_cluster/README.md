## 基于DCU集群的启动方式

为了方便进行大规模训练，我们基于DCU集群，借助slurm进行资源和作业调度管理。在提交训练任务前，需要先在该目录下新建目录：
```
mkdir logs
mkdir hostfile
```
然后使用sbatch命令提交作业脚本以启动训练：
```
sbatch run_dcu.py
```
