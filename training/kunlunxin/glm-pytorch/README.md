### 模型Checkpoint下载
[模型Checkpoint下载](../../benchmarks/glm/README.md#模型checkpoint)
### 测试数据集下载
[测试数据集下载](../../benchmarks/glm/README.md#数据集)

### 昆仑芯XPU配置与运行信息参考
#### 环境配置
- ##### 硬件环境
  - 机器型号: 昆仑芯AI加速器组R480-X8
  - 加速卡型号: 昆仑芯AI加速卡R300
  - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### 软件环境
  - OS版本：Ubuntu 20.04
  - OS kernel版本: 5.4.0-26-generic
  - 加速卡驱动版本：4.0.25
  - Docker镜像和版本：pytorch1.12.1-cpu-ubuntu20.04:v0.01
  - 训练框架版本：xmlir+111e7d45 [xmlir下载](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/archives/111e7d45/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl)
  - 训练编译器版本：xacc+111e7d45 [xacc下载](https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/archives/111e7d45/xacc-0.1.0-cp38-cp38-linux_x86_64.whl)
  - 依赖软件版本：pytorch-1.12.1+cpu

### 测试运行方法

修改`FlagPerf/training/run_benchmarks/config/test_conf.py`文件里的配置项：

```python
VENDOR = "kunlunxin"

ACCE_CONTAINER_OPT = " --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2" + \
    " --device=/dev/xpu3 --device=/dev/xpu4 --device=/dev/xpu5" + \
    " --device=/dev/xpu6 --device=/dev/xpu7 --device=/dev/xpuctrl"

ACCE_VISIBLE_DEVICE_ENV_NAME = "XPU_VISIBLE_DEVICES"

CASES = [
    "GLM_TORCH_DEMO_R300_1X1",
    "GLM_TORCH_DEMO_R300_1X2",
    "GLM_TORCH_DEMO_R300_1X4",
    "GLM_TORCH_DEMO_R300_1X8",
    "GLM_TORCH_DEMO_R300_2X8"
]
```

剩余步骤按照项目根目录文档下的[“快速启动”](../../../README.md#快速启动)章节进行。


### 运行情况参考

| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度 | Steps数 | 性能(samples/s) |
|---------| --------------- | ----------- | -------- | -------- | ------- | ---------------- |
| 单机1卡  | config_R300x1x1 | 121371.25| 0.8 | 0.8021   | 14400(fp32)| 0.50 |
| 单机2卡  | config_R300x1x2 | 106709.60| 0.8 | 0.8085   | 12000(fp32)| 0.92 |
| 单机4卡  | config_R300x1x4 | 44162.12 | 0.8 | 0.8027   | 4800(fp32) | 1.79 |
| 单机8卡  | config_R300x1x8 | 22902.82 | 0.8 | 0.8003   | 2400(fp32) | 3.47 |
| 两机8卡  | config_R300x2x8 | 16217.80 | 0.8 | 0.8012   | 1500(fp32) | 6.08 |

### 许可证

Apache 2.0 license。
