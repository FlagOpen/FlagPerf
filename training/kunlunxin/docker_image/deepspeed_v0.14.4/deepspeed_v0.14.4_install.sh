#!/bin/bash
export https_proxy=http://10.1.0.34:7890
pip install deepspeed==0.14.4

wget https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/R300_plus/latest/xpytorch-cp38-torch201-ubuntu2004-x64.run

bash xpytorch-cp38-torch201-ubuntu2004-x64.run