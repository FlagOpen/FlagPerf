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

set -x

pip install schedule

# xpytorch install
cd /opt/xpytorch && bash xpytorch-cp38-torch201-ubuntu2004-x64.run
CUDART_DUMMY_REGISTER=1 python -m torch_xmlir --doctor &> /tmp/xpytorch.version.out
CUDART_DUMMY_REGISTER=1 python -c "import torch; print(torch.rand(512, 128).cuda())" &> /tmp/xpytorch.test.out
