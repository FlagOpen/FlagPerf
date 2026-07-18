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

import os
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--million_tokens", type=int, default=100)
parser.add_argument("--chatglm3_dir", type=str, default="../chatglm3_6b_hf")
parser.add_argument("--openwebtext_dir",
                    type=str,
                    default="/data/chatglm3_6b_pretrain/openwebtext")
parser.add_argument(
    "--output_file",
    type=str,
    default="/data/chatglm3_6b_pretrain/openwebtext_chatglm3_100M.npy")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.chatglm3_dir, trust_remote_code=True)
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

    tokens = tokenizer.encode_plus(all_text)
    input_ids = tokens['input_ids']
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
