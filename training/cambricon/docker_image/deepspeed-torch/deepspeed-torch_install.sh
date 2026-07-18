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
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install loguru schedule protobuf sentencepiece datasets==2.15.0 schedule==1.2.2 safetensors==0.4.3 numpy==1.26.4
pip3 uninstall -y transformer-engine
# transformers and accelarate
git clone https://gitee.com/xiaoqi25478/cambricon_wheels.git
cd cambricon_wheels/transformers
pip3 install -e .
cd ../accelerate
pip3 install -e .
cd ../../
