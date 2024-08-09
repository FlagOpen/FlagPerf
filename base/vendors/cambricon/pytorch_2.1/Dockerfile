FROM flagperf:cambricon-deepspeed-v24.06-torch2.1.0-catch1.21.0-ubuntu22.04-py310-megatron-patch
#shell
SHELL ["/bin/bash", "-c"]
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends tzdata && apt-get install -y openssh-server && mkdir -p /run/sshd
RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup \
  && sed -i 's|http://.*archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
  && sed -i 's|http://.*security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt update -y && apt install -y sudo dmidecode ipmitool sysstat net-tools sshpass libnuma-dev

# modify ~/.bashrc file
RUN sed -i '/\[ -z "\$PS1" \] \&\& return/s/^/#/' ~/.bashrc
RUN echo -e "\n# Add environment variables\n\
export NEUWARE_HOME=/usr/local/neuware\n\
export LD_LIBRARY_PATH=/usr/local/mpi_wrapper/build/install/lib64:/usr/local/neuware/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH}\n\
export PATH=/torch/venv3/pytorch/bin:/torch/venv3/pytorch/bin:/usr/local/neuware/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}" >> ~/.bashrc
