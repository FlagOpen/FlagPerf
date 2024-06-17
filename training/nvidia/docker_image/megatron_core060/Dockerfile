FROM nvcr.io/nvidia/pytorch:24.02-py3
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN /bin/bash -c "git clone -b core_v0.6.0 https://github.com/NVIDIA/Megatron-LM.git"
