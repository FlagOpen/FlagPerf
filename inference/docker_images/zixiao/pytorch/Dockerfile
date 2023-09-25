FROM amd64/ubuntu:20.04

LABEL VERSION="1.1.9"

ARG DEBIAN_FRONTEND=noninteractive
USER root
WORKDIR /tmp
ARG MOFED_VER=5.4-3.1.0.0
RUN if [ $MOFED_VER ]; then echo "MOFED_VER=$MOFED_VER" & exit 0;else echo "no MOFED_VER, please check" & sleep 1 & exit 1; fi


ARG WHICH_MIRROR=mirrors.cloud.tencent.com/repo

# change source to tencent cloud
RUN sed -i 's#http://archive.ubuntu.com/ubuntu/#http://mirrors.cloud.tencent.com/ubuntu/#g' /etc/apt/sources.list && \
    sed -i 's#http://security.ubuntu.com/ubuntu/#http://mirrors.cloud.tencent.com/ubuntu/#g' /etc/apt/sources.list && \
    apt update && apt -yq install apt-transport-https wget
RUN wget -O /etc/apt/sources.list http://${WHICH_MIRROR}/ubuntu20_sources.list && apt update -yq
RUN mkdir /root/.pip && echo "[global]\nindex- url = https://mirrors.cloud.tencent.com/pypi/simple\ntrusted-host = mirrors.cloud.tencent.com" > /root/.pip/pip.conf

################################ BASIC LIBRARY  #################################
# install packages
RUN apt-get update && apt-get install -yq apt-utils sudo vim curl \
    autoconf automake dialog libtool pkg-config libffi-dev \
    libexpat1-dev libpciaccess-dev libxml2-dev \
    bison flex xutils-dev zlib1g-dev ninja-build git locate \
    zip unzip g++ \
    # install ssh
    openssh-server openssh-client \
    # fix hexdump missing issue
    bsdmainutils \
    # fix header missing for tensorflow verbs support
    libibverbs-dev \
    #install default python3.8
    python3 python3-pip python3-dev python3-tk libjpeg-dev \
    # RMA dependency library
    graphviz dpatch swig gfortran chrpath tk tcl libnl-3-200 libnl-route-3-dev lsof \
    libnl-3-dev libgfortran5 libnl-route-3-200 ethtool libnuma1 libnuma-dev udev \
    # ECCL dependency library
    libncurses5-dev hwloc libhwloc-dev libhwloc-common libboost-all-dev libevent-dev python2-dev && \
    apt-get clean


# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' >/etc/timezone

# Install miniconda
# Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
# RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
RUN wget -O ~/miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /root/miniconda && \
    rm ~/miniconda.sh && \
    /root/miniconda/bin/conda config --set show_channel_urls yes && \
    /root/miniconda/bin/conda create --name python38 python=3.8 -y && \
    /root/miniconda/bin/conda clean -ya

# hyperparamer, typing_extensions, numpy requests
RUN /root/miniconda/envs/python38/bin/pip install \
    --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    hyperparameter \
    typing_extensions \
    numpy \
    requests \
    onnx \
    onnxruntime \
    attrs \
    regex \
    decorator \
    loguru \
    schedule \
    munch \
    pyyaml \
    tqdm \
    scipy

RUN /root/miniconda/envs/python38/bin/pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu

ENV PATH /root/miniconda/envs/python38/bin:$PATH
ENV PATH=/usr/local/zx-smi/zx-smi-1.20.0:$PATH



