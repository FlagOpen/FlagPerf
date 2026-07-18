# Copyright 2026 FlagOS Contributors
#
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
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as t


class ToFloat16(object):

    def __call__(self, tensor):
        return tensor.to(dtype=torch.float16)


def build_train_dataset(config):
    normalize = t.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    traindir = os.path.join(config.data_dir, config.train_data)
    if config.fp16:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            t.Compose([
                t.RandomResizedCrop(224),
                t.RandomHorizontalFlip(),
                t.ToTensor(),
                ToFloat16(), normalize
            ]))
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            t.Compose([
                t.RandomResizedCrop(224),
                t.RandomHorizontalFlip(),
                t.ToTensor(), normalize
            ]))
    return dataset


def build_eval_dataset(config):
    normalize = t.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    evaldir = os.path.join(config.data_dir, config.eval_data)
    if config.fp16:
        dataset = torchvision.datasets.ImageFolder(
            evaldir,
            t.Compose([
                t.Resize(256),
                t.CenterCrop(224),
                t.ToTensor(),
                ToFloat16(), normalize
            ]))
    else:
        dataset = torchvision.datasets.ImageFolder(
            evaldir,
            t.Compose(
                [t.Resize(256),
                 t.CenterCrop(224),
                 t.ToTensor(), normalize]))
    return dataset


def build_train_dataloader(dataset, config):
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True)
    return data_loader


def build_eval_dataloader(dataset, config):
    if config.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=True)
    return data_loader
