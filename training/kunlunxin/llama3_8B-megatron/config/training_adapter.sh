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

echo "[Prompt] kunlunxin adaption start"

TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --fp16 \
    --initial-loss-scale 16384
"

VENDOR_ARGS=" \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn \
    --disable-bias-linear \
    --hidden-dropout 0 --attention-dropout 0 \
    --no-async-tensor-model-parallel-allreduce --no-gradient-accumulation-fusion
"
