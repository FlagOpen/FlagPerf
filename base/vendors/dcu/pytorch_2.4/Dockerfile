FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10
RUN /bin/bash -c "source /opt/dtk/env.sh"
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN /bin/bash -c "pip install loguru"
RUN /bin/bash -c "apt-get update"
RUN /bin/bash -c "apt-get install sshpass"