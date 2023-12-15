#!/bin/bash

set -xe

pip install colorama -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install regex -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pybind11==2.9.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wheel==0.37.1  -i https://pypi.tuna.tsinghua.edu.cn/simple


wget -O ~/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl    https://bd.bcebos.com/klx-pytorch-ipipe-bd/docker_conda_dep/torch201_cuda117/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl
pip install ~/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl


wget  -O ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run  https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/201_cuda/latest/xmlir-cp38-torch201-ubuntu2004-x64-installer.run
bash ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run
rm ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run

pip install psutil==5.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tabulate==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


