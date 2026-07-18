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
set -x

export XMLIR_F_XPU_ENABLED_BOOL=true
export XMLIR_TORCH_XCCL_ENABLED=true

##===----------------------------------------------------------------------===##
## R480 config
##===----------------------------------------------------------------------===##

# BKCL
topo_file=${WORKSPACE-"."}/topo.txt
touch topo_file
export XPUSIM_TOPOLOGY_FILE=$(readlink -f $topo_file)

## workaround due to ccix bug
export BKCL_PCIE_RING=1
export ALLREDUCE_ASYNC="0"
export ALLREDUCE_FUSION="0"
export BKCL_FORCE_SYNC=1

export XACC_ENABLE=1
export XMLIR_D_XPU_L3_SIZE=32505856
