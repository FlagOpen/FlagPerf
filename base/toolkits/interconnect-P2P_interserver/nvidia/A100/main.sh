#!/bin/bash

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

service ssh restart
export NCCL_DEBUG=WARN
export OPAL_PREFIX=/opt/hpcx/ompi
export PATH=/usr/local/nvm/versions/node/v16.20.2/bin:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
nvcc -c -o bandwidth.o bandwidth.cu -I/usr/local/cuda/include -I/usr/local/nccl/include -I/usr/local/mpi/include
mpic++ -o bdtest bandwidth.o -L/usr/local/cuda/lib64 -L/usr/local/nccl/lib -L/usr/local/mpi/lib -lcudart -lnccl -lcuda -lmpi
HOSTS=$(yq '.HOSTS | join(",")' ../../../../configs/host.yaml)
echo "NODERANK: $NODERANK"
if [ "$NODERANK" -eq 0 ]; then
    echo "NODERANK is 0, executing the final command..."
    mpirun --allow-run-as-root --host $HOSTS -np 2 -x NCCL_DEBUG=WARN -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_2,mlx5_5 -x CUDA_DEVICE_MAX_CONNECTIONS=1 ./bdtest
fi
