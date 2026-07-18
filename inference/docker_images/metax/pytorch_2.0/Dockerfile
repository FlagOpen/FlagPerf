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

FROM mxcr.io/library/maca-c500-pytorch:2.19.2.5-ubuntu18.04-amd64
ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTORCH_USE_FLASHATTN=1
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y gcc libgl1-mesa-glx libglib2.0-0

RUN pip install numpy
RUN pip install pyyaml
RUN pip install timm
RUN pip install munch
RUN pip install schedule
RUN pip install loguru
RUN /bin/bash -c "uname -a"
