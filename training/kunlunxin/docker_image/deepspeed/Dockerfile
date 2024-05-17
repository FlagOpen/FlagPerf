FROM iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.27
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python

ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH
