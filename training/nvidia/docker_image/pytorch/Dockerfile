FROM nvcr.io/nvidia/pytorch:21.02-py3
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update -y
RUN apt-get install -y --fix-missing libgl1

