4FROM nvcr.io/nvidia/pytorch:24.03-py3
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN pip3 install loguru
RUN pip3 install pycuda
RUN pip3 install schedule
RUN pip3 install munch