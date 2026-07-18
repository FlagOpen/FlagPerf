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

FROM metax-megatron:2.23.0.13.342-ubuntu20.04-amd64
ENV PATH="/opt/conda/bin:${PATH}"
RUN /bin/bash -c "pip config set global.index-url http://repo.metax-tech.com/r/pypi/simple"

ENV MACA_PATH="/opt/maca"
ENV MACA_CLANG_PATH="${MACA_PATH}/mxgpu_llvm/bin"
ENV MACA_CLANG="${MACA_PATH}/mxgpu_llvm"
ENV DEVINFO_ROOT="${MACA_PATH}"
ENV CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
ENV CUDA_PATH="${CUCC_PATH}"

ENV PATH="${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}"

# ENV GLOO_SOCKET_IFNAME=ens115f0

ENV MACA_SMALL_PAGESIZE_ENABLE=1
ENV MCPYTORCH_DISABLE_PRINT=1

ENV MCCL_NET_GDR_LEVEL=7
ENV MCCL_P2P_LEVEL=SYS
ENV MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
ENV FORCE_ACTIVATE_WAIT=1

ENV SET_DEVICE_NUMA_PREFERRED=1

ENV MAX_JOBS=20
ENV PYTORCH_ENABLE_SAME_RAND_A100=1
ENV MHA_BWD_NO_ATOMIC_F64=1
ENV MCCL_IB_GID_INDEX=1
ENV NVTE_FLASH_ATTN=1
ENV NVTE_FUSED_ATTN=0
ENV MCCL_MAX_NCHANNELS=16

ENV MCBLAS_CUSTOMIZED_CONFIG_PATH=/workspace/Megatron-LM_metax/mcblas_customized_config.yaml

RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
