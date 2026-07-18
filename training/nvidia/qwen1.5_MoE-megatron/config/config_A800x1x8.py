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

batchsize = 1
gbs = 512
seqlength = 8192
padlength = 8192
precision = 'bf16'
tensor_parallel = 1
pipeline_parallel = 2
accumulate_steps = 1
theoryflops = 312000000000000.0
epochs = 1
