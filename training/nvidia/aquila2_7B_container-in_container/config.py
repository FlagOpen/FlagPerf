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

# =========================================================
# network
# =========================================================
SSH_PORT = "60022"

net_cmd = "export CUDA_DEVICE_MAX_CONNECTIONS=1;export NCCL_SOCKET_IFNAME=enp;export NCCL_IB_DISABLE=0;export NCCL_IB_CUDA_SUPPORT=1;export NCCL_IB_GID_INDEX=0;export NCCL_IB_HCA=mlx5_2,mlx5_5;export NCCL_DEBUG=debug"

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "312000000000000"

# =========================================================
# env attribute
# =========================================================
env_cmd = "export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
