FROM registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py310
# RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN apt-get install mpich -y
RUN pip3 install loguru
RUN pip3 install schedule
RUN pip3 install munch
RUN pip3 install argparse
RUN pip3 install pyyaml
RUN echo "ulimit -n 65536" >> ~/.bashrc
RUN echo "ulimit -l unlimited" >> ~/.bashrc