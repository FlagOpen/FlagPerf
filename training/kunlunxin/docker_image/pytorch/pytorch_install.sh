#!/bin/bash

set -xe

wget https://klx-public.bj.bcebos.com/xmlir/flagopen/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl -O ~/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
pip install ~/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
rm ~/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
