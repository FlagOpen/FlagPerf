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

FROM iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.29
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python

ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH
ENV LD_LIBRARY_PATH /workspace/xre-Linux-x86_64-5.0.11.1/so:/workspace/bkcl/so:$LD_LIBRARY_PATH
ENV CUDART_DUMMY_REGISTER 1
ENV XPU_FORCE_USERMODE_LAUNCH 1
ENV XPU_DUMMY_EVENT 1
