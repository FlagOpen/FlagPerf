# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Optional

from torch.utils.data import DataLoader

from .factories import create_dataset_factory
from .feature_spec import FeatureSpec


def get_data_loaders(args, feature_spec: FeatureSpec, device_mapping: Optional[dict] = None) -> \
        Tuple[DataLoader, DataLoader]:
    dataset_factory = create_dataset_factory(args,
                                             feature_spec=feature_spec,
                                             device_mapping=device_mapping)

    dataset_train, dataset_test = dataset_factory.create_datasets()
    train_sampler = dataset_factory.create_sampler(
        dataset_train) if args.shuffle_batch_order else None
    collate_fn = dataset_factory.create_collate_fn()

    data_loader_train = dataset_factory.create_data_loader(
        dataset_train, collate_fn=collate_fn, sampler=train_sampler)
    data_loader_test = dataset_factory.create_data_loader(
        dataset_test, collate_fn=collate_fn)
    return data_loader_train, data_loader_test


def build_train_dataloader(args, feature_spec, device_mapping):
    data_loader_train, _ = get_data_loaders(args, feature_spec, device_mapping)
    return data_loader_train


def build_eval_dataloader(args, feature_spec, device_mapping):
    _, data_loader_eval = get_data_loaders(args, feature_spec, device_mapping)
    return data_loader_eval