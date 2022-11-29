import h5py
import numpy as np
import os

import paddle
from paddle.io import Dataset


class MyDataset(Dataset):
    """
    继承 paddle.io.Dataset 类
    """

    def __init__(self, data_list):

        super(MyDataset, self).__init__()
        self.data_list = data_list

    def __getitem__(self, index):

        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = self.data_list[
            index]

        return input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels

    def __len__(self):
        return len(self.data_list)


class PretrainingDataset(Dataset):

    def __init__(self, input_file, max_pred_length):

        super(PretrainingDataset, self).__init__()
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
        # [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
        #     paddle.to_tensor(input[index].astype(np.int64)) if indice < 5 else paddle.to_tensor(
        #         input[index], dtype=paddle.int64) for indice, input in enumerate(self.inputs)]
        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            input[index].astype(np.int64)
            for indice, input in enumerate(self.inputs)
        ]

        # masked_lm_labels = paddle.zeros(input_ids.shape, dtype=paddle.int64)
        # index = self.max_pred_length
        # masked_token_idx = paddle.nonzero(masked_lm_positions)
        # masked_token_count = paddle.sum(paddle.ones(masked_token_idx.shape, dtype=paddle.int32))
        masked_lm_labels = np.zeros(input_ids.shape, dtype='int64')
        index = self.max_pred_length
        masked_token_idx = np.nonzero(masked_lm_positions)[0]
        masked_token_count = np.sum(
            np.ones(masked_token_idx.shape, dtype='int32'))
        if masked_token_count.item() != 0:
            index = masked_token_count.item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        #   masked_lm_labels[masked_lm_positions[:index].numpy()] = masked_lm_ids[:index]

        return [
            input_ids, segment_ids, input_mask, masked_lm_labels,
            next_sentence_labels
        ]
