FROM nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN pip3 install --extra-index-url https://pypi.nvidia.com tensorrt-llm==0.9.0
RUN pip3 install loguru
RUN pip3 install pycuda
RUN pip3 install schedule
RUN pip3 install munch