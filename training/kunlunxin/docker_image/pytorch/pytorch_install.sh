#!/bin/bash

set -xe

pip install https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip install https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl

pip install psutil==5.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate==0.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tabulate==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m xacc.install
