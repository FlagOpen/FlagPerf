# Copyright 2021 Huawei Technologies Co., Ltd
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
"""data convert"""

import sys

def convert(input_path, output_path):
    data = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            data.append(line[10:-1])
    fp = open(output_path, 'a', encoding='utf-8')
    for name in data:
        fp.write(name+'\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Wrong parameter format.")
        print("Usage:")
        print("    python3 convert.py [SYN__PATH] [NAMES_OUTPUT_PATH_NAME]")
        sys.exit()
    input_p = sys.argv[1]
    output_p = sys.argv[2]
    convert(input_p, output_p)
