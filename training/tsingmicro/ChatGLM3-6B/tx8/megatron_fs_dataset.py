import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


from megatron_core.datasets.indexed_dataset import IndexedDataset
from megatron_core.datasets.gpt_dataset import MockGPTDataset, GPTDataset, GPTDatasetConfig
from megatron_core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron_core.datasets.utils import Split
from megatron_core.datasets.utils import compile_helpers

# TODO, Judge: if RANK == 0:
print("> compiling dataset index builder ...")
compile_helpers()
print(">>> done with dataset index builder ...")



class _Llama3TokenizerFS(MegatronTokenizer):
    def __init__(self, tokenizer_path):
        name = 'HFTokenizer'
        super().__init__(name)
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.eod_id = self.tokenizer.eos_token_id
        self.cls_id = self.tokenizer.bos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self._inv_vocab = None
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + len(self.tokenizer.get_added_vocab())

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        vocab = self.vocab()
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class FlagscaleMegatronDataset(Dataset):
    def __init__(
        self, 
        path_prefix, 
        tokenizer_path,
        max_len=8192, 
        num_samples=None
    ):
        # max_len will affect the sequence order.
        super(FlagscaleMegatronDataset, self).__init__()
        config = GPTDatasetConfig(
            random_seed=42, 
            sequence_length=max_len, 
            blend=([path_prefix], None), 
            blend_per_split=[None, None, None], 
            split='1', 
            tokenizer=_Llama3TokenizerFS(tokenizer_path), 
            reset_position_ids=True, 
            reset_attention_mask=True,
            eod_mask_loss=False
        )
        low_level_dataset = GPTDataset.build_low_level_dataset(path_prefix, config)
        num_elements = GPTDataset.numel_low_level_dataset(low_level_dataset)
        split_indices = np.arange(start=0, stop=num_elements, step=1, dtype=np.int32)
        self.gpt_dataset = GPTDataset(low_level_dataset, path_prefix, split_indices, num_samples, Split.train, config)

        print("FlagscaleMegatronDataset init done...")

    def __len__(self):
        return len(self.gpt_dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(f"FlagscaleMegatronDataset getitem...{i}")
        gpt_data = self.gpt_dataset[i]
        ret = dict(
            input_ids=gpt_data['tokens'],
            labels=gpt_data['labels'],
            attention_mask=gpt_data['loss_mask'].to(torch.bool),
        )
        # print(ret)
        return ret


if __name__ == '__main__':
    # ON 32-A100 server path.
    # path_prefix = "/data/LM/ZhiYuanData/llama_mixtral-tar/SAMPLE50B/llama3/llama3_dataset/dedup-md5-pile-pile-cc_text_document"
    path_prefix = "/workplace/data/hugginface/datasets/flagperf/wudao_llama3bpe_content_document"
    # path_prefix = "/data/flagperf/wudao_llama3bpe_content_document"
    # tokenizer_path = "/data/LM/ZhiYuanData/llama_mixtral-tar/SAMPLE50B/llama3/llama3_tokenizer/"
    tokenizer_path = "/workplace/data/hugginface/models/THUDM--chatglm3-6b"
    dataset = FlagscaleMegatronDataset(path_prefix, tokenizer_path, max_len=1024)
    print(len(dataset))
    
    i = 0
    for data in dataset:
        print(data)
        print("--++"*50)
        i += 1
        if i > 10:
            break
