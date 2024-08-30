FROM cambricon-base/pytorch:v24.04-torch2.3.0-catch1.20.0-ubuntu22.04-py310_20240830
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN pip3 install loguru schedule  munch
ENV FLAGGEMS_WORK_DIR  /workspaces
