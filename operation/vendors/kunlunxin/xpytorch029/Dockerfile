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

#FROM iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.29
FROM xpytorch-flaggems:v0.1
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH
ENV PATH /usr/local/xpu/bin:$PATH
RUN /bin/bash -c 'wget -O /tmp/xre.tar.gz https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.15.1/xre-Linux-x86_64-5.0.15.1.tar.gz && cd /tmp && tar zxf xre.tar.gz && cp -a xre-Linux-x86_64-5.0.15.1 /usr/local/xpu'

#RUN apt-get update
RUN pip3 install loguru
#RUN pip3 install pycuda
RUN pip3 install schedule
RUN pip3 install munch
#RUN /bin/bash -c 'wget -O /tmp/xpytorch.run https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/R300_plus/latest/xpytorch-cp38-torch201-ubuntu2004-x64.run && bash /tmp/xpytorch.run'
#RUN /bin/bash -c 'source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda'

ENV TRITON_XPU_ARCH 3
ENV CUDART_DUMMY_REGISTER 1
ENV XPURT_DISPATCH_MODE PROFILING
