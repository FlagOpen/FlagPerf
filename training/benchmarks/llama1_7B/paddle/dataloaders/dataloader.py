import os

import numpy as np
import paddle

from paddlenlp.utils.log import logger
from .dataset import GPTDataset, get_train_valid_test_split_
def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and "_idx.npz" in str(f))
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

def create_pretrained_dataset(
    data_args,
    training_args,
    data_file,
    tokenizer,
):

    train_valid_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        training_args.per_device_eval_batch_size
        * training_args.dataset_world_size
        * training_args.eval_iters
        * (training_args.max_steps // training_args.eval_steps + 1),
        training_args.per_device_eval_batch_size * training_args.dataset_world_size * training_args.test_iters,
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

    splits = get_train_valid_test_split_(data_args.split, len(sample_lens))
    assert len(sample_lens) >= splits[-1], "The document nums should larger than max of splits, but %s < %s" % (
        len(sample_lens),
        splits[-1],
    )

    def print_dataset(data, mode="train"):
        # logger.info(f"Sample data for {mode} mode")
        input_ids, loss_mask, attention_mask, position_ids, labels = data
        # logger.info(tokenizer._decode(input_ids))
        # logger.info(tokenizer._decode(labels))
        # logger.info(tokenizer.convert_ids_to_tokens(input_ids))

    def build_dataset(index, name):
        dataset = GPTDataset(
            file_prefix=input_prefix,
            build_data_file=training_args.local_process_index == 0,
            micro_batch_size=training_args.per_device_train_batch_size
            if name == "train"
            else training_args.per_device_eval_batch_size,
            name="gpt_" + name,
            max_seq_len=data_args.max_seq_length,
            num_samples=train_valid_test_num_samples[index],
            documents=np.arange(splits[index], splits[index + 1]),
            sample_ids=sample_ids,
            sample_lens=sample_lens,
            eos_id=tokenizer.eos_token_id,
            seed=training_args.seed,
        )
        print_dataset(dataset[0], name)
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
