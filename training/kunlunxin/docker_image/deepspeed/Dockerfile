# TODO: this is a temporary docker image from Docker Hub, need update to kunlunxin's Harbor registry.
FROM 127.0.0.1:9999/xpytorch/kunlunxin-deepspeed:v1.0

RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"

WORKDIR /workspace

ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH
