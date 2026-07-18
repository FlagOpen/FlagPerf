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

import math
from driver import dist_pytorch
from fairseq import data
from fairseq.data import data_utils, load_dataset_splits


def build_datasets(args):
    dist_pytorch.main_proc_print('building dataset ...')
    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)
    return datasets


def build_train_dataloader(datasets, args):
    """Training dataloaders."""
    dist_pytorch.main_proc_print('building train dataloaders ...')
    train_dataloader = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )
    return train_dataloader


def build_valid_dataloader(datasets, args):
    """valid dataloaders."""
    dist_pytorch.main_proc_print('building valid dataloaders ...')
    valid_dataset = data.EpochBatchIterator(
        dataset=datasets[args.valid_subset],
        max_tokens=None,
        max_sentences=max(8, min(math.ceil(1024/args.distributed_world_size), 128)),
        max_positions=args.max_positions,
        required_batch_size_multiple=8,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    return valid_dataset


def build_test_dataloader(datasets, args):
    """test dataloaders."""
    dist_pytorch.main_proc_print('building test dataloaders ...')
    test_dataset = data.EpochBatchIterator(
        dataset=datasets[args.gen_subset],
        max_tokens=None,
        max_sentences=max(8, min(math.ceil(1024 / args.distributed_world_size), 128)),
        max_positions=args.max_positions,
        required_batch_size_multiple=8,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    return test_dataset


def build_dataloader(args):
    # 加载数据集
    args.data = args.data_dir
    datasets = build_datasets(args)
    # 初始化dataloader
    train = build_train_dataloader(datasets, args)
    valid = build_valid_dataloader(datasets, args)
    test = build_test_dataloader(datasets, args)
    return train, valid, test
