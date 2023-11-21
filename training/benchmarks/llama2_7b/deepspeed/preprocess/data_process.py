import os
from transformers import LlamaTokenizer
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--million_tokens", type=int, default=100)
parser.add_argument("--llama2_dir", type=str, default="../llama2_7b_hf")
parser.add_argument("--openwebtext_dir",
                    type=str,
                    default="/data/llama2_7b_pretrain/openwebtext")
parser.add_argument(
    "--output_file",
    type=str,
    default="/data/llama2_7b_pretrain/openwebtext_llama2_100M.npy")
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.llama2_dir)
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
