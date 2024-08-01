FROM iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.29
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python

ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH
ENV LD_LIBRARY_PATH /workspace/xre-Linux-x86_64-5.0.11.1/so:/workspace/bkcl/so:$LD_LIBRARY_PATH
ENV CUDART_DUMMY_REGISTER 1
ENV XPU_FORCE_USERMODE_LAUNCH 1
ENV XPU_DUMMY_EVENT 1