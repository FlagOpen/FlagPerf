FROM nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update

COPY tesorrt-llm_install.sh /tesorrt-llm_install.sh
RUN chmod +x /tesorrt-llm_install.sh && /tesorrt-llm_install.sh

RUN pip3 install loguru
RUN pip3 install schedule
RUN pip3 install munch