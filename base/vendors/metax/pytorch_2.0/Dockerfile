FROM llama2_70b_qwen_0720:2.23.0.13.342-ubuntu20.04-amd64
ENV PATH=$PATH:/opt/conda/bin
RUN /bin/bash -c "pip3 config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN pip3 install loguru
RUN pip3 install schedule
RUN pip3 install argparse
RUN pip3 install pyyaml
ENV MCCL_FAST_WRITE_BACK=1
ENV MCCL_EARLY_WRITE_BACK=15
ENV MCCL_P2P_LEVEL=SYS
ENV MCCL_NET_GDR_LEVEL=SYS
ENV MCCL_CROSS_NIC=1
