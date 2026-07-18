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

# for 1*8 3150
# VENDOR_ARGS=" \
#     --transformer-impl local  \
#     --use-distributed-optimizer \
#     --use-mcore-models \
#     --use-flash-attn \
#     --pipline-num-layers-list 7 9 9 7
# "
# for 4*8
VENDOR_ARGS=" \
    --transformer-impl local  \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --pipline-num-layers-list 16 16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1 \
    --recompute-num-layers-list 5 0
"
