FROM registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4
RUN /bin/bash -c "uname -a"
RUN pip3 uninstall -y pylint
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --upgrade typing-extensions
RUN git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP && git checkout 513f763ec7dc50f7d52a332b61266a0bb860f10e && pip3 install . && cd ..