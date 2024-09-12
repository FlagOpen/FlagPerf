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

# 在容器内创建 .ssh 目录
RUN mkdir -p /root/.ssh

# 将公钥和私钥复制到容器内
COPY id_rsa /root/.ssh/id_rsa
COPY id_rsa.pub /root/.ssh/id_rsa.pub

# 设置正确的权限
RUN chmod 600 /root/.ssh/id_rsa
RUN chmod 644 /root/.ssh/id_rsa.pub

# 添加公钥到 authorized_keys 文件以实现免密登录
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys