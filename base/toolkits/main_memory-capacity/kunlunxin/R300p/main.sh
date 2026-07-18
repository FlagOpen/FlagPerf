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

TOOL=xpu-smi
LOG=_${TOOL}.log.${RANDOM}.$$
PERF=/opt/xre/bin/$TOOL

mem=$($PERF -m | head -1 | awk '{print $19}')
echo "[FlagPerf Result] main_memory-capacity=$mem MiB"
sleep 360
