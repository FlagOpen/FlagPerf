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

export https_proxy=http://10.1.0.34:7890
pip install deepspeed==0.11.1

wget https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/zhangling21_baichuan2/xmlir_fixeq.run

bash xmlir_fixeq.run
XFLAGS --disable megatron_23_05
