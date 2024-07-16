FROM flagperf-operation-metax-ngctorch2403:v1.0
RUN /bin/bash -c "pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV MACA_VISIBLE_DEVICES=7
RUN apt-get update
RUN pip3 install loguru
#RUN pip3 install pycuda
RUN pip3 install schedule
RUN pip3 install munch
