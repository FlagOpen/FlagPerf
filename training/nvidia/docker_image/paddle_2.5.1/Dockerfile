FROM registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.2-cudnn8.2-trt8.0
RUN /bin/bash -c "uname -a"
RUN pip3 uninstall -y pylint
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --upgrade typing-extensions
RUN pip3 install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html