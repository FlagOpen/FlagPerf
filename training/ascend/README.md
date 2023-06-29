# 厂商信息

官网: https://www.hiascend.com/zh/

昇腾计算产业生态包括围绕着昇腾计算技术和产品体系所开展的学术、技术、公益及商业活动，产生的知识和产品以及各种合作伙伴，主要包括原始设备制造商OEM、原始设计制造商ODM、 独立硬件开发商IHV、咨询与解决方案集成商C&SI、独立软件开发商ISV、云服务提供商XaaS等。
同时，昇腾的合作伙伴体系当中还包含围绕昇腾相关产品对外提供服务交付的服务类伙伴，提供培训服务的人才联盟伙伴，提供投融资和运营服务的投融资运营伙伴等。
昇腾高度重视高校的人才培养和昇腾开发者的发展，让高校和开发者成为整个昇腾生态的重要组成部分。

# FlagPerf适配验证环境说明
## 环境配置参考
  - 硬件
    - 机器型号： 
    - 多机网络类型、带宽: RoCE，100 Gb/s
  - 软件
    - OS版本：Ubuntu 18.04.1 LTS
    - OS kernel版本: Linux 4.15.0-29-generic aarch64
    - Docker 版本: 19.03.6

## 容器镜像信息
- 容器构建信息
  - Dockerfile路径：training/ascend/docker_image/\<framework\>/Dockerfile
  - 构建后软件安装脚本: training/ascend/docker_image/\<framework\>/\<framework\>_install.sh

- 核心软件信息

  - AI框架&版本
    - mindspore: 1.10.0

  - 其它软件版本
    - cann: 6.3.RC1


## 加速卡监控采集
- 加速卡使用信息采集命令

  ```bash
  npu-smi info
  ```
- 监控项示例：
    ```bash
+----------------------------------------------------------------------------------------------------+
| npu-smi 22.0.3                            Version: 22.0.3                                          |
+-------------------+-----------------+--------------------------------------------------------------+
| NPU     Name      | Health          | Power(W)          Temp(C)              Hugepages-Usage(page) |
| Chip    Device    | Bus-Id          | AICore(%)         Memory-Usage(MB)                           |
+===================+=================+==============================================================+
| 4       910A     | OK              | NA                43             0          / 970            |
| 0       0         | 0000:81:00.0    | 0                 861  / 21534                               |
+===================+=================+==============================================================+
    ```
- 加速卡使用信息采集项说明

|监控项| 日志文件 |
|---|---|---|
|Temp(C) | ascend_monitor.log |
|AICore(%) |ascend_monitor.log |
|Memory-Usage(MB) |ascend_monitor.log |
|Hugepages-Usage(page) |ascend_monitor.log |
|Power(W) |ascend_monitor.log |