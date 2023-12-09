#!/bin/bash

set -xe

wget  -O ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run  https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/201_cuda/latest/xmlir-cp38-torch201-ubuntu2004-x64-installer.run
bash ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run
rm ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run

pip install psutil==5.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate==0.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tabulate==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


