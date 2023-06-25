# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Create SimQAT algorithm instance."""

from mindspore_gs import SimulatedQuantizationAwareTraining as SimQAT


def create_simqat():
    algo = SimQAT()
    algo.set_act_quant_delay(900)
    algo.set_weight_quant_delay(900)
    algo.set_act_symmetric(False)
    algo.set_weight_symmetric(True)
    algo.set_act_per_channel(False)
    algo.set_weight_per_channel(True)
    algo.set_enable_fusion(True)
    algo.set_bn_fold(True)
    algo.set_one_conv_fold(False)
    return algo
