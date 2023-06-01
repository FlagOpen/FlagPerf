from torch.utils.data import DataLoader
import copy
from common.fairseq.data import data_utils
<<<<<<< HEAD
from common.helpers import print_once
from common.sampler import DistributedIndicesSampler
import numpy as np

def build_train_dataloader(
        dataset,
        training,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        num_concat_batches=1,
=======
from common.sampler import DistributedIndicesSampler
import numpy as np


def build_train_dataloader(
    dataset,
    training,
    max_tokens=None,
    max_sentences=None,
    max_positions=None,
    ignore_invalid_inputs=False,
    required_batch_size_multiple=1,
    seed=1,
    num_shards=1,
    shard_id=0,
    num_workers=0,
    num_concat_batches=1,
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
):
    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()
    # filter examples that are too large
    if max_positions is not None:
<<<<<<< HEAD
        indices = filter_indices_by_size(
            indices, dataset, max_positions, ignore_invalid_inputs)
=======
        indices = filter_indices_by_size(indices, dataset, max_positions,
                                         ignore_invalid_inputs)
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

    # create mini-batches with given size constraints
    batch_inds, non_grouped_batch_inds = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
        num_concat_batches=num_concat_batches,
    )

    batch_ids = copy.deepcopy(non_grouped_batch_inds)
    [bi.fill(i) for i, bi in enumerate(batch_ids)]
    inds_ids = zip(np.concatenate(batch_inds), np.concatenate(batch_ids))
    dataset.batch_ids = {idx: batch_idx for idx, batch_idx in inds_ids}

    # Batches are already specified, now we just need to shuffle them
<<<<<<< HEAD
    batch_ind_sampler = DistributedIndicesSampler(batch_inds, shuffle=training,
                                                  num_replicas=num_shards,
                                                  rank=shard_id, seed=seed,
=======
    batch_ind_sampler = DistributedIndicesSampler(batch_inds,
                                                  shuffle=training,
                                                  num_replicas=num_shards,
                                                  rank=shard_id,
                                                  seed=seed,
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
                                                  drop_last=training,
                                                  fillvalue=[])
    loader = DataLoader(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_ind_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return loader, batch_ind_sampler

<<<<<<< HEAD
build_eval_dataloader = build_train_dataloader



# def build_eval_dataloader(
#         dataset,
#         training,
#         max_tokens=None,
#         max_sentences=None,
#         max_positions=None,
#         ignore_invalid_inputs=False,
#         required_batch_size_multiple=1,
#         seed=1,
#         num_shards=1,
#         shard_id=0,
#         num_workers=0,
#         num_concat_batches=1,
# ):
#     # get indices ordered by example size
#     with data_utils.numpy_seed(seed):
#         indices = dataset.ordered_indices()

#     # filter examples that are too large
#     if max_positions is not None:
#         indices = filter_indices_by_size(
#             indices, dataset, max_positions, ignore_invalid_inputs)

#     # create mini-batches with given size constraints
#     batch_inds, non_grouped_batch_inds = dataset.batch_by_size(
#         indices,
#         max_tokens=max_tokens,
#         max_sentences=max_sentences,
#         required_batch_size_multiple=required_batch_size_multiple,
#         num_concat_batches=num_concat_batches,
#     )

#     batch_ids = copy.deepcopy(non_grouped_batch_inds)
#     [bi.fill(i) for i, bi in enumerate(batch_ids)]
#     inds_ids = zip(np.concatenate(batch_inds), np.concatenate(batch_ids))
#     dataset.batch_ids = {idx: batch_idx for idx, batch_idx in inds_ids}

#     # Batches are already specified, now we just need to shuffle them
#     batch_ind_sampler = DistributedIndicesSampler(batch_inds, shuffle=training,
#                                                   num_replicas=num_shards,
#                                                   rank=shard_id, seed=seed,
#                                                   drop_last=training,
#                                                   fillvalue=[])
#     loader = DataLoader(
#         dataset=dataset,
#         collate_fn=dataset.collater,
#         batch_sampler=batch_ind_sampler,
#         num_workers=num_workers,
#         pin_memory=True,
#         persistent_workers=num_workers > 0,
#     )

#     return loader, batch_ind_sampler

def filter_indices_by_size(
    indices, dataset, max_positions=None, ignore_invalid_inputs=False
):
=======

build_eval_dataloader = build_train_dataloader


def filter_indices_by_size(indices,
                           dataset,
                           max_positions=None,
                           ignore_invalid_inputs=False):
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
    """
    Filter examples that are too large

    Args:
        indices (np.array): original array of sample indices
        dataset (~fairseq.data.FairseqDataset): dataset to batch
        max_positions (optional): max sentence length supported by the
            model (default: None).
        ignore_invalid_inputs (bool, optional): don't raise Exception for
            sentences that are too long (default: False).
    Returns:
        np.array: array of filtered sample indices
    """
    indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
    # TODO: consider removing this function. If `len(ignored) > 0`,
    #  an error is raised in fairseq dataset code, both in sup and unsup case
    if len(ignored) > 0:
        if not ignore_invalid_inputs:
            raise Exception(
<<<<<<< HEAD
                (
                    "Size of sample #{} is invalid (={}) since max_positions={}, "
                    "skip this example with --skip-invalid-size-inputs-valid-test"
                ).format(ignored[0], dataset.size(ignored[0]), max_positions)
            )
        print(
            (
                "WARNING: {:,} samples have invalid sizes and will be skipped, "
                "max_positions={}, first few sample ids={}"
            ).format(len(ignored), max_positions, ignored[:10])
        )
    return indices
=======
                ("Size of sample #{} is invalid (={}) since max_positions={}, "
                 "skip this example with --skip-invalid-size-inputs-valid-test"
                 ).format(ignored[0], dataset.size(ignored[0]), max_positions))
        print(("WARNING: {:,} samples have invalid sizes and will be skipped, "
               "max_positions={}, first few sample ids={}").format(
                   len(ignored), max_positions, ignored[:10]))
    return indices
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
