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

FROM llama2_70b_qwen_0720:2.23.0.13.342-ubuntu20.04-amd64
ENV PATH=$PATH:/opt/conda/bin
RUN /bin/bash -c "pip3 config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN pip3 install loguru
RUN pip3 install schedule
RUN pip3 install argparse
RUN pip3 install pyyaml
ENV MCCL_FAST_WRITE_BACK=1
ENV MCCL_EARLY_WRITE_BACK=15
ENV MCCL_P2P_LEVEL=SYS
ENV MCCL_NET_GDR_LEVEL=SYS
ENV MCCL_CROSS_NIC=1
