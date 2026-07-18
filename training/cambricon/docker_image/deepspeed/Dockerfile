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

FROM flagperf:cambricon-v24.08.02-torch2.1.0-catch1.22.1-ubuntu22.04-py310-megatron-patch
#shell
SHELL ["/bin/bash", "-c"]
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends tzdata && apt-get install -y openssh-server && mkdir -p /run/sshd
RUN apt update -y && apt install -y sudo dmidecode ipmitool sysstat net-tools sshpass
# modify ~/.bashrc file
RUN sed -i '/\[ -z "\$PS1" \] \&\& return/s/^/#/' ~/.bashrc
RUN echo -e "\n# Add environment variables\n\
export NEUWARE_HOME=/usr/local/neuware\n\
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH}\n\
export PATH=/torch/venv3/pytorch/bin:/usr/local/neuware/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}\n\
export CNCL_MLULINK_OVER_ROCE_DISABLE=1\n\
export CNCL_MLULINK_CROSS_HOSTS_ENABLE=0\n\
export CNCL_MLU_DIRECT_LEVEL=1" >> ~/.bashrc
