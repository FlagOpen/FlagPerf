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

FROM 10.1.15.74:5000/zhiyuan_flagperf_ai_platform:v2.0

ENV CUDART_DUMMY_REGISTER 1
ENV XPU_DUMMY_EVENT 1

ENV XPU_ZEBU_MODE 1
ENV BKCL_XLINK_C2C 1
ENV XBLAS_FC_HBM_VERSION 40
ENV XPU_FORCE_CODE_PARAM_LOCATE_IN_L3 1
ENV XPU_FORCE_USERMODE_LAUNCH 1

ENV LD_LIBRARY_PATH /usr/local/xpu/xre/so:/usr/local/xpu/xccl/so:$LD_LIBRARY_PATH
ENV XMLIR_FA_GEMM_TYPE float16
ENV FAST_SWIGLU_ENABLE 1
ENV XDNN_FAST_DIV_SCALAR true
ENV FAST_SUB_MN_M true

RUN /bin/bash -c "git clone https://github.com/FlagOpen/FlagScale -b release/v0.2"
RUN /bin/bash -c "git clone https://github.com/FlagOpen/FlagPerf.git -b main"
