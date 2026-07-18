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

echo "[Prompt] iluvatar adaption is not NULL, for other Vendors"
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=4
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
VENDOR_ARGS=" \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --no-create-attention-mask-in-dataloader \
    --use-legacy-models \
    --num-layers-per-stage 1 7 2 9 1 7 \
    --disable-bias-linear \
"
