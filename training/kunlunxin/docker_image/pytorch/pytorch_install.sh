#!/bin/bash


# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xe

pip install https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip install https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/latest/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl

pip install psutil==5.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate==0.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tabulate==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m xacc.install
