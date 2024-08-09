#!/bin/bash
set -xe
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install regex==2024.5.15 schedule==1.2.2 accelerate==0.31.0 transformers==4.40.1 pybind11