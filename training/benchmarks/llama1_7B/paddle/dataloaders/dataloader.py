import os

import numpy as np
import paddle

from paddlenlp.utils.log import logger
from paddlenlp.data.causal_dataset import build_train_valid_test_datasets, print_rank_0


def create_pretrained_dataset(
    data_args,
    training_args,
    data_file,
    tokenizer,
    need_data=True,
):
    train_val_test_num_samples = [
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

    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl=data_args.data_impl,
        splits_string=data_args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_length,
        seed=training_args.seed,
        skip_warmup=data_args.skip_warmup,
        data_cache_path=data_args.data_cache,
        need_data=need_data,
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode.")
        # input_ids, loss_mask, attention_mask, position_ids, labels = data
        input_ids = data["text"]

        logger.info(tokenizer._decode(input_ids))

    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        return {
            "input_ids": tokens,
            "labels": labels,
        }

    # if need_data:
    #     print_dataset(train_dataset[0], "train")
    #     print_dataset(valid_dataset[0], "valid")
    #     print_dataset(test_dataset[0], "test")
    return train_dataset, valid_dataset, test_dataset, _collate_data

def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and ("_idx.npz" in str(f) or ".idx" in str(f)))
        ]
        files = [x.replace("_idx.npz", "") for x in files]
        files = [x.replace(".idx", "") for x in files]  # add

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files