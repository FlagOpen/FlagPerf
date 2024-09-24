#!/bin/bash
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

