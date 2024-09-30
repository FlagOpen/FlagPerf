FROM klx-flagperf-mcore060:latest
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH

RUN pip3 install loguru
RUN pip3 install schedule
RUN pip3 install munch
