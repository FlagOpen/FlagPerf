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

pip3 install ./packages/TopsInference-2.4.12-py3.8-none-any.whl

dpkg -i ./sdk_installers/topsruntime_2.4.12-1_amd64.deb
dpkg -i ./sdk_installers/tops-sdk_2.4.12-1_amd64.deb
