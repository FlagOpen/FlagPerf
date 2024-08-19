#!/bin/bash

set -x

# conda env
source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
pip install pytest loguru schedule

# xpytorch install
#wget -q -O xpytorch.run https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/R300_plus/latest/xpytorch-cp38-torch201-ubuntu2004-x64.run && bash xpytorch.run &> install-xpytorch.log
CUDART_DUMMY_REGISTER=1 python -m torch_xmlir --doctor
CUDART_DUMMY_REGISTER=1 python -c "import torch; print(torch.rand(512, 128).cuda())"

