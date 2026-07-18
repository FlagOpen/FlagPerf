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

export GLOG_V=3
export HCCL_CONNECT_TIMEOUT=600

if [-f /usr/local/Ascend/nnae/set_env.sh];then
    source /usr/local/Ascend/nnae/set_env.sh
elif [-f /usr/local/Ascend/ascend-toolkit/set_env.sh];then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [-f ~/Ascend/nnae/set_env.sh];then
    source ~/Ascend/nnae/set_env.sh
elif [-f ~/Ascend/ascend-toolkit/set_env.sh];then
    source ~/Ascend/ascend-toolkit/set_env.sh
else
    echo "warning find no env so not set"
fi
