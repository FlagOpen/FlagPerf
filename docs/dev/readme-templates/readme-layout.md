# FlagPerf文档层级说明

## README目录层级

docs/dev/     规范文档主目录

run_pretraining.example.py  # run_pretraining模版

├── readme-templates  *# readme模版主目录*
│   ├── case-readme-template.md               *# case readme*
│   ├── model-readme-template.md 		   *# 模型readme*
│   ├── readme-layout.md                  			*# readme层级结构* 
│   ├── vendor-case-readme-template.md  *# 厂商case readme*
│   └── vendor-readme-template.md          *# 厂商readme*
└── specifications  *# 规范文档主目录*
    ├── case-adatpion-spec.md    					 # 厂商适配case规范
    └── standard-case-spec.md 					   # 标准case规范



Repo README：repo根目录下，一般无需修改

模型 README: training/benchmark/<model>下，每个模型一个文档

标准Case README：training/benchmark/<model>-<framework>下，每个case一个文档

厂商 README: training/<vendor>/下，每个vendor一个文档，向用户介绍厂商信息，说明适配FlagPerf测试Case的软、硬件环境信息及加速卡监控采集指标

厂商适配case README: training/<vendor>/<model>-<framework>下，产商的每个case一个文档

```Bash
├── LICENSE.md
├── README.md   # REPO README
├── docs
│   └── dev
│       ├── case_adaptation.md          # case适配规范 
│       ├── case_creation.md            # case添加规范 
│       └── run_pretraining.example.py  # run_pretraining模版
├── flagperf_wechat.png
├── logo.png
└── training
    ├── benchmarks
    │   ├── driver
    │   ├── bert
    │   │   ├── README.md
    │   │   └── paddle
    │   │       ├── ……
    │   │       └── readme.md
    │   ├── cpm
    │   ├── glm
    │   ├── mobilenetv2
    │   └── resnet50
    ├── nvidia
    │   ├── README.md
    │   ├── bert-paddle
    │   │   ├── README.md
    │   │   └── config
    │   ├── cpm-pytorch
    │   │   ├── README.md
    │   │   ├── config
    │   │   └── extern
    │   ├── docker_image
    │   ├── glm-pytorch
    │   │   ├── README.md
    │   │   ├── config
    │   │   └── extern
    │   ├── mobilenetv2-pytorch
    │   └── nvidia_monitor.py
    ├── requirements.txt
    ├── run_benchmarks
    └── utils
```

## 总结

代码中需要带的README文档如下：

```Bash
training/benchmarks/<model>
    ├── 模型 README
    ├── <framework>
            ├── 标准Case README
training/<vendor>
    ├── 厂商 README
    ├── <model>-<framework>
            ├── 厂商适配case README（按需可选）
```