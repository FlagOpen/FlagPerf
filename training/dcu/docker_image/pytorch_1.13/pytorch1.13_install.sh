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

proxy_server_ip=10.0.35.251
PROXY_URL="http://$proxy_server_ip:3128/"
NO_PROXY_ADDR="127.0.0.1,localhost,.local,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export no_proxy="$NO_PROXY_ADDR"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export NO_PROXY="$NO_PROXY_ADDR"
