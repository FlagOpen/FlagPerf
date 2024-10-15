#!/bin/bash

set -x

# cd /opt/xpytorch && bash xpytorch-cp39-torch201-ubuntu2004-x64.run

CUDART_DUMMY_REGISTER=1 python -m torch_xmlir --doctor &> /tmp/xpytorch.version.out
CUDART_DUMMY_REGISTER=1 python -c "import torch; print(torch.rand(512, 128).cuda())" &> /tmp/xpytorch.test.out
/etc/init.d/ssh restart
