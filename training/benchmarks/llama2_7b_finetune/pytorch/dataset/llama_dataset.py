import datasets
from itertools import chain
from dataclasses import dataclass

from transformers import LlamaTokenizer


def get_llama_dataset(train_config):
    model_path = train_config.data_dir + "/" + train_config.model_name

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
    })
    dataset_config = generate_dataset_config()
    path = train_config.data_dir + train_config.dataset_dir
    dataset_train = get_preprocessed_dataset(
        train_config,
        path,
        tokenizer,
        dataset_config,
        split="train",
    )

    dataset_val = get_preprocessed_dataset(
        train_config,
        path,
        tokenizer,
        dataset_config,
        split="test",
    )

    return dataset_train, dataset_val, tokenizer


class Concatenator(object):

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size]
                    for i in range(0, chunk_num *
                                   self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


def get_preprocessed_samsum(train_config, path, tokenizer, split):
    # dataset = datasets.load_dataset("samsum", split=split)
    dataset = datasets.load_dataset(path=path, split=split)
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text":
            prompt.format(
                dialog=sample["dialogue"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template,
                          remove_columns=list(dataset.features))

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(chunk_size=train_config.seq_length), batched=True)
    return dataset


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 512


def generate_dataset_config():
    dataset_config = samsum_dataset()
    return dataset_config


def get_preprocessed_dataset(
                             train_config,
                             path,
                             tokenizer,
                             dataset_config,
                             split: str = "train"):

    def get_split():
        return (dataset_config.train_split
                if split == "train" else dataset_config.test_split)

    return get_preprocessed_samsum(
        train_config,
        path,
        tokenizer,
        get_split(),
    )
