# coding=utf-8
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

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
import pdb
from typing import Optional
import numpy as np
import paddle
from .dataset import GPTDataset
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, DistributedBatchSampler
from icecream import ic

def get_train_data_file(config):
    input_dir = os.path.join(config.base_path, config.data_dir)
    if len(input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return input_dir.split()
    else:
        files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if (os.path.isfile(os.path.join(input_dir, f)) and "_idx.npz" in str(f))
        ]
        files = [x.replace("_idx.npz", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret
    return files

def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""
    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index

def create_pretrained_dataset(
    config,
    data_file,
    tokenizer,
):

    train_valid_test_num_samples = [
        config.per_device_train_batch_size
        * config.dataset_world_size
        * config.max_steps
        * config.gradient_accumulation_steps,
        config.per_device_eval_batch_size
        * config.dataset_world_size
        * config.eval_iters
        * (config.max_steps // config.eval_steps + 1),
        config.per_device_eval_batch_size * config.dataset_world_size * config.test_iters,
    ]

    input_prefix = data_file[0]

    for suffix in ["_ids.npy", "_idx.npz"]:
        if not os.path.isfile(input_prefix + suffix):
            raise ValueError("File Not found, %s" % (input_prefix + suffix))

    sample_ids = np.load(input_prefix + "_ids.npy", mmap_mode="r", allow_pickle=True)
    # All documment ids, extend as 1-D array.

    process_data = np.load(input_prefix + "_idx.npz")
    # The len(sample_lens) num of docs
    # The sum(sample_lens) should equal len(sample_ids)
    sample_lens = process_data["lens"]

    splits = get_train_valid_test_split_(config.split, len(sample_lens))

    assert len(sample_lens) >= splits[-1], "The document nums should larger than max of splits, but %s < %s" % (
        len(sample_lens),
        splits[-1],
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode")
        input_ids, loss_mask, attention_mask, position_ids, labels = data
        logger.info(tokenizer._decode(input_ids))
        # logger.info(tokenizer._decode(labels))
        # logger.info(tokenizer.convert_ids_to_tokens(input_ids))

    def build_dataset(index, name):
        dataset = GPTDataset(
            file_prefix=input_prefix,
            build_data_file=config.local_process_index == 0,
            micro_batch_size=config.per_device_train_batch_size
            if name == "train"
            else config.per_device_eval_batch_size,
            name="gpt_" + name,
            max_seq_len=config.max_seq_length,
            num_samples=train_valid_test_num_samples[index],
            documents=np.arange(splits[index], splits[index + 1]),
            sample_ids=sample_ids,
            sample_lens=sample_lens,
            eos_id=tokenizer.eos_token_id,
            seed=config.seed,
        )
        
        # print_dataset(dataset[0], name)
        return dataset

    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # 0:input_ids, 1:loss_mask, 2:attention_mask, 3:position_ids, 4:labels
        for i in (0, 1, 2, 3, 4):
            out[i] = stack_fn([x[i] for x in data])

        return {
            "input_ids": out[0],
            # "token_type_ids": out[1],
            # "attention_mask": out[2],
            # "loss_mask": out[3],
            "labels": out[4],
        }

    # Note, data should be broardcast to all devices.
    # for train, valid, test, the distinct data num is data_world_size
    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return train_dataset, valid_dataset, test_dataset, _collate_data

def _get_train_sampler(config,train_dataset) -> Optional[paddle.io.Sampler]:
    # if config.world_size <= 1:
    #     return paddle.io.BatchSampler(
    #         dataset=train_dataset,
    #         shuffle=True,
    #         batch_size=config.per_device_train_batch_size,
    #         drop_last=config.dataloader_drop_last,
    #     )
    

    return DistributedBatchSampler(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        num_replicas=config.dataset_world_size,
        rank=paddle.distributed.get_rank(),
        drop_last=config.dataloader_drop_last
    )

def _get_eval_sampler(config, eval_dataset):
    # if config.world_size <= 1:
    #     return paddle.io.BatchSampler(
    #         eval_dataset,
    #         batch_size=config.per_device_eval_batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #     )
    # else:
        # drop_last = False
        # if config.pipeline_parallel_degree > 1:
        #     drop_last = True
        #     logger.warning(
        #         "In parallel mode, the bacth_size is strictly checked. set DistributedBatchSampler drop_last=True."
        #     )

    return DistributedBatchSampler(
        eval_dataset,
        num_replicas=config.dataset_world_size,
        rank=paddle.distributed.get_rank(),
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        drop_last=config.dataloader_drop_last,
    )


def load_data(config, tokenizer):
    data_file = get_train_data_file(config)
    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
        config, data_file, tokenizer
    )

    train_sampler = _get_train_sampler(config, train_dataset)
    eval_sampler = _get_eval_sampler(config, eval_dataset)
    test_sampler = _get_eval_sampler(config, test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=config.num_workers,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        collate_fn=data_collator,
        num_workers=config.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=data_collator,
        num_workers=config.num_workers,
    )

    return train_dataloader, eval_dataloader, test_dataloader