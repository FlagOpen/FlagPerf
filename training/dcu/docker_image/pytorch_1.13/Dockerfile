FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.13.1-centos7.6-dtk23.10-py38
RUN source /opt/dtk/env.sh
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
