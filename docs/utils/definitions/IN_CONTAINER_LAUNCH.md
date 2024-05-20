# 容器内启动评测简述

To ensure environmental stability, eliminate the interference caused by different baremetal machines, and provide possible reproducibility, FlagPerf automatically creates images and runs containers for each benchmarking case, executing the specific benchmarking case within the container. However, this benchmarking process has several dependencies on the physical machine environment, such as:

1. Users must have permission to log in to the baremetal machine.
2. The baremetal machine must be able to locally use Docker to create containers.
3. Password-free SSH needs to be configured between baremetal machines.

In some scenarios, it is difficult to meet these dependencies. Therefore, starting from PR [542](https://github.com/FlagOpen/FlagPerf/pull/542), FlagPerf supports "IN_CONTAINER_LAUNCH". When using IN_CONTAINER_LAUNCH mode:

1. FlagPerf will no longer build images and start containers; instead, it will use pre-created containers.
2. FlagPerf will start monitoring within the container, although this approach may involve some risks.
3. Users only need permission to log in to the container and do not need the baremetal machine to be able to locally build containers (e.g., using Kubernetes to create containers in batches).
4. Users must ensure password-free SSH between all container nodes and ensure the container environments are consistent and functional. FlagPerf will not check the container environment and will not ensure that the container environment matches the DockerFile and various shell scripts described in the benchmarking cases.

为了确保环境稳定、排除不同物理机带来的环境干扰、提供可能的复现，FlagPerf在每个评测样例中均自动制作镜像并启动容器，在容器内执行具体的评测任务。这一评测流程对物理机环境有若干依赖，例如

  1. 用户具有登录到物理机的权限
  2. 物理机能够本地使用docker创建容器
  3. 物理机间需要配置ssh免密

  在一些场景中，上述依赖总是不容易达成的。因此FlagPerf从[542](https://github.com/FlagOpen/FlagPerf/pull/542)开始，支持《容器内启动》，具体来说：

  1. FlagPerf不再执行构建镜像、启动容器等过程，而是使用预先创建好的容器
  2. FlagPerf将在容器内启动监控，尽管此方式可能面临风险
  3. 用户仅需要具有登录到容器的权限，且不需要物理机能够本地构建容器（例如使用k8s批量创建容器）
  4. 用户需要确保所有节点的容器间ssh免密，并确保各容器环境统一、正常。FlagPerf将不会对容器环境做检查，不会确保容器环境与评测样例对应DockerFile及各类sh脚本所描述环境相同。

# 用户操作指南

## 设置环境变量

```
export EXEC_IN_CONTAINER=True
```

## 确保容器内硬件驱动、网络、硬件虚拟化等服务器基础配置齐全

1. 确保可连中国大陆可访问网站，速率正常
2. 确保容器镜像、容器内软件包对应版本安装正确
3. 确保可在容器内找到硬件
4. 确保各服务器间root帐号的ssh信任关系和sudo免密
5. 确保monitor相关工具已安装:包括cpu(sysstat)、内存(free)、功耗(ipmitool)、系统信息(加速卡状态查看命令)。例如ubuntu系统中，使用apt install [sysstat/ipmitool]安装

# 特性声明

当用户设置环境变量EXEC_IN_CONTAINER=True时，即表示FlagPerf运行于Docker镜像中。与在物理机上运行相比，程序运行工作流在执行时会跳过以下4步：
- 登录到所有节点，准备镜像
- 在所有节点，启动容器
- 在所有节点，配置容器环境
- 在所有节点，关闭容器
基于上述设计，用户在使用容器内启动的特性时，配置Docker环境，Docker间ssh免密登录，对应case库依赖的步骤交由用户完成。