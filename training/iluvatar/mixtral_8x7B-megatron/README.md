### Iluvatar GPU配置与运行信息参考
#### 环境配置
- ##### BI-V150硬件环境
    - 机器型号: R5300 G5 
    - 加速卡型号: Iluvatar Bi-150
    - CPU型号: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
    - 多机网络类型、带宽: InfiniBand，200Gb/s

- ##### BI-V150软件环境
   - OS版本：Ubuntu 20.04 LTS
   - OS kernel版本: 5.4.0-148-generic   
   - 加速卡驱动版本：3.4.0
   - Docker 版本：20.10.25
   - 训练框架版本：megatron-lm 0.6.0+corex.3.4.0.20240531.104
   - 依赖软件版本：transformers==4.37.1,wandb>=0.15.7,hydra-core 


- ##### 并行策略

   - 并行技术：张量、流水、数据混合并行，具体并行方案见“运行情况”章节
   - 实施者：megatron

- ##### 优化策略

   - transformer-engine impl


### 运行情况

* 输入批尺寸
  1. local_batchsize(micro_batchsize)，简写为LBS，即实际进入模型的张量批尺寸，为config_BI-V150x1x16.py中所写，在本case中默认为1。**厂商适配时可任意更改**
  2. seqlength(max_position_embedding)，简写为MPE，即实际进入模型的序列长度，在本case中默认为2048，原则上不可更改
* 具体操作
  1. 获取run包：corex-docker-installer-3.4.0.20240531.113-10.2-ubuntu20.04-py3.10-x86_64.run，放置位置/FlagPerf/training/iluvatar/docker_image/megatron/sdk_installers  ###由于是daily-sdk不能直接制作需要手动完成，需要先bash *.run得到一个docker image【flagperf-iluvatar-megatron:t_v0.1】
  安装方式：1、accept；2、点击"install driver";3、点击"set image name",改为flagperf-iluvatar-megatron:t_v0.1；4、点击"install"
  2. 获取mixtral_8x7B的运行代码，放置位置/FlagPerf/data_dir,>联系邮箱: contact-us@iluvatar.com  ###也可以放置在其他位置需要修改/FlagPerf/training/iluvatar/mixtral_8x7B-megatron/config/config_BI-V150x1x16.py中mixtral_iluvatar_path的位置；根据自己机器修改同级training_adapter.sh中MASTERADDR的ip。
  3. 由于算法引用的层级不一致，需要修改/FlagPerf/training/benchmarks/mixtral_8x7B/megatron/run_pretraining.py第63行origin_file = os.path.join(megapath, "megatron/megatron/training/arguments.py")和origin_file = os.path.join(megapath, "megatron/megatron/training/tokenizer/tokenizer.py")；修改megatron_main.sh中run_cmd="torchrun $DISTRIBUTED_ARGS $MEGAPATH/megatron/pretrain_gpt.py 
  4. /FlagPerf/training/run_benchmarks/config/test_conf.py，下载pip库的源需要用清华源"https://pypi.tuna.tsinghua.edu.cn/simple/";再执行python3 ./run_benchmarks/run.py
  5. 单机测试中/FlagPerf/training/iluvatar/mixtral_8x7B-megatron/config/config_BI-V150x1x16.py：tensor_parallel=2，pipeline_parallel=2；/FlagPerf/training/iluvatar/mixtral_8x7B-megatron/config/training_adapter.sh：num-layers=8.四机测试中/FlagPerf/training/iluvatar/mixtral_8x7B-megatron/config/config_BI-V150x1x16.py：tensor_parallel=4，pipeline_parallel=2；/FlagPerf/training/iluvatar/mixtral_8x7B-megatron/config/training_adapter.sh：num-layers=32.
  注意：若出现卡断现象，先停掉所有进程执行"ixsmi -r"

* 通用指标

| 指标名称    | 指标值                   | 特殊说明                                     |
| ------- | --------------------- | ---------------------------------------- |
| 任务类别    | 自然语言理解                |                                          |
| 模型      | mixtral_8*7B             |                                          |
| 数据集     | wudao                 | wudao数据集来源于智源研究院<br>bin/idx数据集文件来源于阿里云灵骏团队<br>使用llama tokenizer预处理 |
| 数据精度    | precision,见“性能指标”     | 可选bf16                     |
| 超参修改    | parallel,见“性能指标”      | 格式为PPxDPyTPz，例如PP2DP4TP1                 |
| 超参修改    | fix_hp,见“性能指标”        | 跑满硬件设备评测吞吐量所需特殊超参, global batchsize=1200                        |
| 硬件设备简称  | nvidia H100           |                                          |
| 硬件存储使用  | mem,见“性能指标”           | 通常称为“显存”,单位为GiB                          |
| 计算使用率   | MFU,见“性能指标”           | 参见PaLM论文定义                               |
| **吞吐量** | **token/p/s,见“性能指标”** | 平均单卡每秒处理的token数                          |

* 性能指标

精度对齐需第21步及之后，所有步的loss与nvidia对应步的loss平均相对误差小于2%。NVloss曲线请联系智源研究院获取
#目前仅支持单机
| 配置             | precision | parallel  | fix_hp | token/p/s | 是否精度对齐     | mem   | MFU         |
| -------------- | --------- | --------- | ------ | --------- | ---------- | ----- | ----------- |
| BI150单机8卡（1x8）  | bf16   | PP2DP4EP4TP2 | / | 20106.0  | *（仅供性能参考）* | 41/64 | 7.98%       |
