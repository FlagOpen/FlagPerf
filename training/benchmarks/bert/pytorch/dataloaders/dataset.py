import h5py
import numpy as np
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


class PretrainingDataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else
            torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long)
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids, segment_ids, input_mask, masked_lm_labels,
            next_sentence_labels
        ]


def exchange_padding_fast(device, max_batch_size, input_ids, segment_ids,
                          input_mask, masked_lm_labels, next_sentence_labels):
    #torch.cuda.nvtx.range_push('exchangepadding')
    pad_size = max_batch_size - input_ids.shape[0]
    if pad_size > 0:
        input_ids = F.pad(input_ids, (0, 0, 0, pad_size))
        segment_ids = F.pad(segment_ids, (0, 0, 0, pad_size))
        input_mask = F.pad(input_mask, (0, 0, 0, pad_size))
        masked_lm_labels = F.pad(masked_lm_labels, (0, 0, 0, pad_size))
        next_sentence_labels = F.pad(next_sentence_labels, (0, pad_size))

    ngpus = 1
    igpu = 0
    if dist_pytorch.is_dist_avail_and_initialized():
        ngpus = dist_pytorch.get_world_size()
        igpu = dist_pytorch.get_rank()
    nseqs = input_mask.shape[0]
    ntokensperseq = input_mask.shape[1]

    flattened_length_seq = nseqs * ntokensperseq
    flattened_length_nsp = nseqs

    def get_local_packet_size():
        return 4 * flattened_length_seq + flattened_length_nsp

    # Storing tensors in same order as arguments
    def encode_packet(input_ids, segment_ids, input_mask, masked_lm_labels,
                      next_sentence_labels):

        packet = torch.zeros([get_local_packet_size()],
                             device=device,
                             dtype=torch.int16)

        curr_pos = 0

        packet[curr_pos:curr_pos +
               flattened_length_seq] = input_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos +
               flattened_length_seq] = segment_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos +
               flattened_length_seq] = input_mask.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos +
               flattened_length_seq] = masked_lm_labels.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos +
               flattened_length_nsp] = next_sentence_labels.view(-1)[:]

        return packet

    def decode_packet(flat_packet):
        packet = flat_packet.view(ngpus, get_local_packet_size())

        curr_pos = 0

        input_ids_ = packet[:, curr_pos:curr_pos +
                            flattened_length_seq].contiguous().view(
                                ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        segment_ids_ = packet[:, curr_pos:curr_pos +
                              flattened_length_seq].contiguous().view(
                                  ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        input_mask_ = packet[:, curr_pos:curr_pos +
                             flattened_length_seq].contiguous().view(
                                 ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        masked_lm_labels_ = packet[:, curr_pos:curr_pos +
                                   flattened_length_seq].contiguous().view(
                                       ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        next_sentence_labels_ = packet[:, curr_pos:curr_pos +
                                       flattened_length_nsp].contiguous().view(
                                           ngpus, nseqs)

        return input_ids_, segment_ids_, input_mask_, masked_lm_labels_, next_sentence_labels_

    tensors = encode_packet(input_ids, segment_ids, input_mask,
                            masked_lm_labels, next_sentence_labels)

    tensors_ = torch.zeros([ngpus, get_local_packet_size()],
                           device=device,
                           dtype=torch.float16)
    tensors_ = list(torch.split(tensors_, 1))
    # Address valueError: ProcessGroupGloo::allgather: invalid tensor size at index 0 (expected (2049), got (1, 2049))

    if dist_pytorch.is_dist_avail_and_initialized():
        dist.all_gather(tensors_, tensors.view(torch.float16).unsqueeze(0))
    else:
        tensors_ = tuple(tensors.view(torch.float16).unsqueeze(0))

    tensors_ = torch.stack(tensors_).view(torch.int16).long()
    input_ids_, segment_ids_, input_mask_, masked_lm_labels_, next_sentence_labels_ = decode_packet(
        tensors_)

    seqlens_, indices = torch.sort(input_mask_.sum(dim=2).view(-1),
                                   descending=True)

    if pad_size > 0:
        input_ids_sorted = input_ids_.view(ngpus * nseqs,
                                           ntokensperseq)[indices[:], :]
        segment_ids_sorted = segment_ids_.view(ngpus * nseqs,
                                               ntokensperseq)[indices[:], :]
        input_mask_sorted = input_mask_.view(ngpus * nseqs,
                                             ntokensperseq)[indices[:], :]
        masked_lm_labels_sorted = masked_lm_labels_.view(
            ngpus * nseqs, ntokensperseq)[indices[:], :]
        next_sentence_labels_sorted = next_sentence_labels_.view(
            ngpus * nseqs)[indices[:]]
        # we need to remove the empty sequences we added to the batch
        valid_idx = seqlens_.view(nseqs, ngpus)[:, igpu] > 0
        input_ids_sorted = input_ids_sorted.view(
            nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        segment_ids_sorted = segment_ids_sorted.view(
            nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        input_mask_sorted = input_mask_sorted.view(
            nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_sorted.view(
            nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_sorted.view(
            nseqs, ngpus)[valid_idx, igpu].contiguous()
    else:
        indices_ = indices.view(nseqs, ngpus)[:, igpu]
        input_ids_sorted = input_ids_.view(
            nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        segment_ids_sorted = segment_ids_.view(
            nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        input_mask_sorted = input_mask_.view(
            nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_.view(
            nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_.view(
            nseqs * ngpus)[indices_].contiguous()

    #torch.cuda.nvtx.range_pop()
    return input_ids_sorted, segment_ids_sorted, input_mask_sorted, masked_lm_labels_sorted, next_sentence_labels_sorted
