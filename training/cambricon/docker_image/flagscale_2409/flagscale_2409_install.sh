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

#!/bin/bash
set -xe
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install regex==2024.5.15 schedule==1.2.2 accelerate==0.31.0 transformers==4.40.1 protobuf==3.20.0
pip3 install pybind11 hydra-core s3fs braceexpand webdataset wandb loguru sentencepiece datasets
pip3 install megatron-energon==2.2.0

# 配置免密
echo 'root:123456' | chpasswd
sed -i '/StrictHostKeyChecking/c StrictHostKeyChecking no' /etc/ssh/ssh_config
sed -i 's/#Port 22/Port 55623/g' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
/etc/init.d/ssh restart
sleep 10
sshpass -p "123456" ssh-copy-id -i /root/.ssh/id_rsa.pub -p 55623 root@`hostname -i | awk '{print $1}'`
