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

git clone https://github.com/alibaba/Pai-Megatron-Patch.git
cd /workspace/Pai-Megatron-Patch
git checkout aa7c56272cb53a7aeb7fa6ebbfa61c7fa3a5c2e4
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
git submodule init
git submodule update Megatron-LM-240405
