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

echo "MTHREADS PLACEHOLDER ENV.SH"
export MCCL_PROTOS=2
export MCCL_TOPO_ENHANCE_PLUGIN=None
export MCCL_ALGOS=1
export MUSA_BLOCK_DISTRIBUTION_GRANULARITY=1
export MUSA_EXECUTE_COUNT=1
export MCCL_BUFFSIZE=41943040
export MCCL_IB_GID_INDEX=3
export MUSA_EXECUTION_TIMEOUT=1000000
export MCCL_GRAPH_FILE=/usr/local/musa/topo
export MCCL_IB_HCA='=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1'
