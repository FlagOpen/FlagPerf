FROM flagperf_ascend:v2025

SHELL ["/bin/bash", "-c"]
RUN echo "192.168.1.1 node-name" /etc/hosts

ENV EXEC_IN_CONTAINER True
ENV LD_LIBRARY_PATH /usr/local/python3.11.5/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/python3.11.5/bin:$PATH
ENV PYTHONPATH /usr/local/python3.11.5/lib/python3.11/site-packages:$PYTHONPATH

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh
RUN source /usr/local/Ascend/nnal/atb/set_env.sh
RUN source /usr/local/Ascend/toolbox/set_env.sh

ENV LD_LIBRARY_PATH /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.11.5/lib:/usr/local/Ascend/driver/tools/hccn_tool/:/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/dcmi
