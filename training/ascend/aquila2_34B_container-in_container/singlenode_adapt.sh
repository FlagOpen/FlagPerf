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

TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --sequence-parallel \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --no-shared-fs \
    --use-flash-attn \
    --npu-fa-pre-tokens 65536 \
    --npu-fa-next-tokens 0 \
    --npu-fa-shape-order SBH \
    --use-npu-swiglu \
    --device-type ascend \
    --log-interval 1 \
    --distributed-timeout-minutes 120
"
