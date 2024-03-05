FROM mxcr.io/library/maca-c500-pytorch:2.19.2.5-ubuntu18.04-amd64
ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTORCH_USE_FLASHATTN=1
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy
RUN pip install pyyaml
RUN pip install timm
RUN pip install munch
RUN pip install schedule
RUN pip install loguru
RUN /bin/bash -c "uname -a"