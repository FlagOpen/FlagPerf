FROM mxcr.io/library/maca-c500-pytorch:2.19.2.5-ubuntu18.04-amd64
ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTORCH_USE_FLASHATTN=1
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y gcc libgl1-mesa-glx libglib2.0-0

RUN pip install numpy
RUN pip install pyyaml
RUN pip install timm
RUN pip install munch
RUN pip install schedule
RUN pip install loguru
RUN pip install accelerate
RUN /bin/bash -c "uname -a"