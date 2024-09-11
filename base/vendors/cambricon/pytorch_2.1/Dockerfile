FROM flagperf:cambricon-v24.08.02-torch2.1.0-catch1.22.1-ubuntu22.04-py310-megatron-patch
#shell
SHELL ["/bin/bash", "-c"]
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends tzdata && apt-get install -y openssh-server && mkdir -p /run/sshd
RUN apt update -y && apt install -y sudo dmidecode ipmitool sysstat net-tools sshpass libnuma-dev

# modify ~/.bashrc file
RUN sed -i '/\[ -z "\$PS1" \] \&\& return/s/^/#/' ~/.bashrc
RUN echo -e "\n# Add environment variables\n\
export NEUWARE_HOME=/usr/local/neuware\n\
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH}\n\
export PATH=/torch/venv3/pytorch/bin:/usr/local/neuware/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}" >> ~/.bashrc
