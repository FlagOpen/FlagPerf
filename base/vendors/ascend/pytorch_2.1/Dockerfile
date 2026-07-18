# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM flagperf_ascend:v2025

SHELL ["/bin/bash", "-c"]

ENV EXEC_IN_CONTAINER True
ENV LD_LIBRARY_PATH /usr/local/python3.11.5/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/python3.11.5/bin:$PATH
ENV PYTHONPATH /usr/local/python3.11.5/lib/python3.11/site-packages:$PYTHONPATH

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh
RUN source /usr/local/Ascend/nnal/atb/set_env.sh
RUN source /usr/local/Ascend/toolbox/set_env.sh

ENV LD_LIBRARY_PATH /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.11.5/lib:/usr/local/Ascend/driver/tools/hccn_tool/:/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/dcmi
