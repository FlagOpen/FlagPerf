FROM cambricon_flagscale_2409_pt21
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
