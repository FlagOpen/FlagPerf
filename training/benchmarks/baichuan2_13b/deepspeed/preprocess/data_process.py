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

import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
# # 获取 DEEPSPEED 目录的绝对路径
DEEPSPEED_PATH = '/root/DEEPSPEED'  # 请根据实际情况替换为 DEEPSPEED 的路径
# # 将 DEEPSPEED 路径添加到 sys.path
sys.path.append(DEEPSPEED_PATH)
# # 现在 DEEPSPEED 应该已经被添加到 sys.path 中
# # 可以导入 DEEPSPEED 目录下的模块或包了
from baichuan_13b_hf.tokenization_baichuan import BaichuanTokenizer
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('Baichuan-13B-Base')

os.environ['TMPDIR'] = '/mnt/ruanhongji'

parser = argparse.ArgumentParser()
parser.add_argument("--million_tokens", type=int, default=100)
parser.add_argument("--baichuan_dir", type=str, default="../baichuan_13b_hf")
parser.add_argument("--openwebtext_dir",
                    type=str,
                    default="../../openwebtext")
parser.add_argument(
    "--output_file",
    type=str,
    default="../openwebtext_baichuan_100M.npy")
args = parser.parse_args()

tokenizer = BaichuanTokenizer(vocab_file="../baichuan_13b_hf/tokenizer.model")
tokenizer.pad_token = tokenizer.eos_token

dir_path = args.openwebtext_dir
file_list = [
    os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
]

all_tokens = np.array([], dtype=int)
write_buffer = np.array([], dtype=int)

iters = 0

for file_path in tqdm(file_list):
    all_text = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        all_text += f.read()

    tokens = tokenizer._tokenize(all_text)
    input_ids = tokenizer._convert_token_to_id(tokens)

    write_buffer = np.append(write_buffer, np.array(input_ids, dtype=int))

    if len(write_buffer) > 1000000:
        all_tokens = np.append(all_tokens, write_buffer)
        write_buffer = np.array([], dtype=int)

    if iters % 1000 == 0:
        print("Tokens num: ", len(all_tokens))

    if len(all_tokens) > 1000000 * args.million_tokens:
        break

    iters += 1

np.save(args.output_file, all_tokens)
