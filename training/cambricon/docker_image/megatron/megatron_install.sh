#!/bin/bash
set -xe
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install regex==2024.5.15 schedule==1.2.2 accelerate==0.31.0 transformers==4.40.1 pybind11
