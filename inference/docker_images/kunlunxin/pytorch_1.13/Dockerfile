FROM ubuntu:18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        wget \
        git \
        locales \
        locales-all \
        python3.8-dev \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*


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

RUN cd /root && wget https://baidu-kunlun-public.su.bcebos.com/XTCL/XTCL-2.1/XTCL-ubuntu_x86_64.tar.gz && tar -xzf XTCL-ubuntu_x86_64.tar.gz

ENV LD_LIBRARY_PATH=/root/XTCL-ubuntu_x86_64/3rdparty/lib:/root/XTCL-ubuntu_x86_64/runtime/shlib:/root/XTCL-ubuntu_x86_64/shlib
ENV KERNEL_INCLUDE=/root/XTCL-ubuntu_x86_64/xpu/kernels
ENV XTCL_L3_SIZE=67104768
ENV XPU_PADDLE_L3_SIZE=67104768
ENV RT_LIBRARY_PATH=/root/XTCL-ubuntu_x86_64/runtime/shlib
ENV THIRDPARTY_LIB_DIR=/root/XTCL-ubuntu_x86_64/3rdparty/lib
ENV XTCL_INSTALL_DIR=/root/XTCL-ubuntu_x86_64
ENV XTCL_QUANTIZE_WEIGHT=1
ENV XTCL_USE_FP16=1
ENV PYTHONPATH=/root/XTCL-ubuntu_x86_64/python:/root/XTCL-ubuntu_x86_64/python/tvm:/root/XTCL-ubuntu_x86_64/python/topi
ENV CLANG_PATH=/root/XTCL-ubuntu_x86_64
ENV KERNEL_SEARCH_PATH=/root/XTCL-ubuntu_x86_64/xpu/kernels
ENV XPUSIM_DEVICE_MODEL=KUNLUN2
ENV XTCL_AUTO_ALLOC_L3=1
ENV TVM_DIR=/root/XTCL-ubuntu_x86_64



ENV PATH /root/miniconda/envs/python38/bin:$PATH

