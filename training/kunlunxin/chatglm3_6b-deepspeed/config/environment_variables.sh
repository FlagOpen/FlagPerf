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

export LD_LIBRARY_PATH=/workspace/xre-Linux-x86_64-5.0.11.1/so:/workspace/bkcl/so:$LD_LIBRARY_PATH
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export XPU_DUMMY_EVENT=1
export USE_FAST_BF16_FC=True
export XMLIR_FA_GEMM_TYPE=float16
export XMLIR_API_DEFAULT_SIZE=4000000000 
export XBLAS_FC_HBM_VERSION=40
export XDNN_USE_FAST_SWISH=true
export XDNN_FAST_DIV_SCALAR=true
