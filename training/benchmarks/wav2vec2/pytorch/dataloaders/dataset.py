from common.fairseq.data import AddTargetDataset, FileAudioDataset
from pathlib import Path


# Supervised CTC training
class LabelEncoder(object):

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(label,
                                           append_eos=False,
                                           add_if_not_exist=False)


def build_train_dataset(subset,
                        config,
                        target_dictionary=None,
                        with_labels=False,
                        training=True):

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
        with open(Path(config.data,
                       f"{config.train_subset}.{config.labels}")) as f:
            labels = [line for i, line in enumerate(f) if i not in skip_inds]

        assert len(labels) == len(dataset), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(dataset)}) do not match")

        dataset = AddTargetDataset(
            dataset,
            labels,
            pad=target_dictionary.pad(),
            eos=target_dictionary.eos(),
            batch_targets=True,
            process_label=LabelEncoder(target_dictionary),
            add_to_input=False)

    return dataset


build_eval_dataset = build_train_dataset
