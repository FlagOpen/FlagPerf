#!/bin/bash

set -xe

source activate python38_torch201_cuda

wget  -O ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run  https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/201_cuda/xmlir-cp38-torch201-ubuntu2004-x64-installer.run && \
    bash ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run && \
    rm ~/xmlir-cp38-torch201-ubuntu2004-x64-installer.run

wget -O ~/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl    https://bd.bcebos.com/klx-pytorch-ipipe-bd/docker_conda_dep/torch201_cuda117/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl && \
    /root/miniconda/envs/python38_torch201_cuda/bin/pip  install ~/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl && \
    rm ~/torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl

wget -O ~/deepspeed-0.11.1.0-py3-none-any.whl  https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/deepspeed/latest/deepspeed-0.11.1.0-py3-none-any.whl && \
    /root/miniconda/envs/python38_torch201_cuda/bin/pip  install ~/deepspeed-0.11.1.0-py3-none-any.whl && \
    rm ~/deepspeed-0.11.1.0-py3-none-any.whl
