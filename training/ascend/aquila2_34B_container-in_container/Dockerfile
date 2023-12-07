FROM ascend-aquila2:v1
RUN /bin/bash -c "git clone https://github.com/FlagOpen/FlagScale -b release/v0.2"
RUN /bin/bash -c "git clone https://github.com/FlagOpen/FlagPerf.git -b AI_platform"
RUN /bin/bash -c "pip install -e FlagScale/ascend/megatron_npu"