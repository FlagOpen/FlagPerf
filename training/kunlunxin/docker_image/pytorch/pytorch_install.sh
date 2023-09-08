#!/bin/bash

set -xe

pip install https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip install https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl

python -m xacc.install
