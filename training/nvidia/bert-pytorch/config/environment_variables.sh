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

# =================================================
# Export variables
# =================================================

export CONTAINER_MOUNTS="--gpus all"
NVCC_ARGUMENTS="-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr -ftemplate-depth=1024 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80"
if [[ "$PYTORCH_BUILD_VERSION" == 1.8* ]]; then
    NVCC_ARGUMENTS="${NVCC_ARGUMENTS} -D_PYTORCH18"
fi
export NVCC_ARGUMENTS
