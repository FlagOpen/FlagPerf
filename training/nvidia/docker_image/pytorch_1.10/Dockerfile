# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.10-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt .
RUN export https_proxy=http://10.1.0.34:7890
RUN export http_proxy=http://10.1.0.34:7890
RUN export all_proxy=socks5h://10.1.0.34:7891


WORKDIR /workspace/dlrm

COPY . .

RUN chmod +x bind.sh
# RUN pip install --no-cache-dir -e .