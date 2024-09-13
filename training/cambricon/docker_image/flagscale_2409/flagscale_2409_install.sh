#!/bin/bash
#!/bin/bash
set -xe
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install regex==2024.5.15 schedule==1.2.2 accelerate==0.31.0 transformers==4.40.1 
pip3 install pybind11 hydra-core s3fs braceexpand webdataset wandb loguru sentencepiece
pip3 install megatron-energon==2.2.0
#配置免密
sed -i '/StrictHostKeyChecking/c StrictHostKeyChecking no' /etc/ssh/ssh_config
sed -i 's/#Port 22/Port 9876/g' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
/etc/init.d/ssh restart
