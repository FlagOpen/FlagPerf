#!/bin/bash

set -x

pip install pytest loguru schedule

/etc/init.d/ssh start

# xpytorch install
cd /opt/xpytorch && bash xpytorch-cp38-torch201-ubuntu2004-x64.run
CUDART_DUMMY_REGISTER=1 python -m torch_xmlir --doctor
CUDART_DUMMY_REGISTER=1 python -c "import torch; print(torch.rand(512, 128).cuda())"

