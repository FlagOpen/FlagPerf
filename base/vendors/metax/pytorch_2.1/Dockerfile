FROM mxc500-torch2.1-py310:mc2.32.0.3-ubuntu22.04-amd64
ENV PATH=$PATH:/opt/conda/bin
RUN /bin/bash -c "pip3 config set global.index-url  https://pypi.tuna.tsinghua.edu.cn/simple"
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
