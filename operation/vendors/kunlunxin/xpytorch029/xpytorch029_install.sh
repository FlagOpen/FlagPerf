#!/bin/bash

set -x

# conda env
source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
pip install pytest loguru schedule

# xpytorch install
wget -q -O xpytorch.run https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/R300_plus/latest/flaggems/xpytorch-cp38-torch201-ubuntu2004-x64.run && bash xpytorch.run &> install-xpytorch.log
CUDART_DUMMY_REGISTER=1 python -m torch_xmlir --doctor
CUDART_DUMMY_REGISTER=1 python -c "import torch; print(torch.rand(2,3).cuda())"

# xpu triton
wget -q -O triton-2.1.0-cp38-cp38-linux_x86_64.whl https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/R300_plus/latest/flaggems/triton-2.1.0-cp38-cp38-linux_x86_64.whl && pip install --no-deps --force-reinstall ./triton-2.1.0-cp38-cp38-linux_x86_64.whl &> install-triton.log
pip show triton
cp -v /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/triton/testing.py /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/triton/testing.py.bak
wget -q -O /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/triton/testing.py https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/R300_plus/latest/flaggems/triton.testing.py


# FlagGems
test -d FlagGems && mv FlagGems FlagGems.bak
git clone https://mirror.ghproxy.com/https://github.com/FlagOpen/FlagGems.git
#git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
git checkout v2.0-perf-klx
pip install -e . --no-deps


# test flaggems
export TRITON_XPU_ARCH=3
export CUDART_DUMMY_REGISTER=1
cd /home/FlagGems && python -m pytest -s tests/test_binary_pointwise_ops.py::test_accuracy_add[dtype0-0.001-shape0]
