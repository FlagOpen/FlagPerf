<<<<<<< HEAD
import wav2vec2.arg_parser
from common.fairseq.data import AddTargetDataset, FileAudioDataset
from common.utils import AttrDict, print_once
from wav2vec2.model import Wav2Vec2Model, Wav2VecEncoder, Wav2VecCtc
from pathlib import Path
import copy
import numpy as np

# Supervised CTC training
class LabelEncoder(object):
=======
from common.fairseq.data import AddTargetDataset, FileAudioDataset
from pathlib import Path


# Supervised CTC training
class LabelEncoder(object):

>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
<<<<<<< HEAD
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )
    

def build_train_dataset(subset, config, target_dictionary=None, with_labels=False,
                 training=True):
=======
        return self.dictionary.encode_line(label,
                                           append_eos=False,
                                           add_if_not_exist=False)


def build_train_dataset(subset,
                        config,
                        target_dictionary=None,
                        with_labels=False,
                        training=True):
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

    dataset = FileAudioDataset(
        manifest_path=Path(config.data_dir, f'{subset}.tsv'),
        sample_rate=config.sample_rate,
        min_sample_size=config.min_sample_size if training else None,
        max_sample_size=config.max_sample_size if training else None,
        pad=(hasattr(config, 'labels') or config.enable_padding),
        normalize=config.normalize,
        num_buckets=config.num_batch_buckets,
        compute_mask_indices=False,
        repeat_to_refsize=(config.num_concat_batches > 1),
    )
    if with_labels:
        assert config.labels
        assert hasattr(config, 'labels')

        skip_inds = getattr(dataset, "skipped_indices", set())
<<<<<<< HEAD
        with open(Path(config.data, f"{config.train_subset}.{config.labels}")) as f:
=======
        with open(Path(config.data,
                       f"{config.train_subset}.{config.labels}")) as f:
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
            labels = [line for i, line in enumerate(f) if i not in skip_inds]

        assert len(labels) == len(dataset), (
            f"labels length ({len(labels)}) and dataset length "
<<<<<<< HEAD
            f"({len(dataset)}) do not match"
        )
=======
            f"({len(dataset)}) do not match")
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

        dataset = AddTargetDataset(
            dataset,
            labels,
            pad=target_dictionary.pad(),
            eos=target_dictionary.eos(),
            batch_targets=True,
            process_label=LabelEncoder(target_dictionary),
<<<<<<< HEAD
            add_to_input=False
        )

    return dataset

=======
            add_to_input=False)

    return dataset


>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
build_eval_dataset = build_train_dataset
