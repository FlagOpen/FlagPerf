# TODO: this is a temporary docker image from Docker Hub, need update to kunlunxin's Harbor registry.
#from dynamicheart/pytorch1.12.1-cpu-ubuntu20.04:v0.01
from  dongfangshenzhu/flagperf-deepspeed:v1.0
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
# This is the original Dockerfile used to build `pytorch1.12.1-cpu-ubuntu18.04:v0.04`.
# The Dockerfile only installs minimal dependencies for kunlunxin, i.e. numpy and pytorch1.12.1
