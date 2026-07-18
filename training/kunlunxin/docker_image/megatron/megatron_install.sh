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

# using github mirrors to avoid github TTL
#export https_proxy=http://10.1.0.34:7890
git clone https://githubfast.com/FlagOpen/FlagScale
cd FlagScale

git checkout eb0438a5459404e2e4c70b15fa37e9a197ab159d
echo 'export PYTHONPATH=$PYTHONPATH:/home/FlagScale' >> /root/.bashrc
source /root/.bashrc

wget https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/zhangling21_llama70B/xmlir201_5.run
bash xmlir201_5.run
XFLAGS --enable transformer_engine
XFLAGS --enable flagscale
