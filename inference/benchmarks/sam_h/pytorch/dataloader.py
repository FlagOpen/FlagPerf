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

import torchvision as tv
from torch.utils.data import DataLoader as dl
from torch.utils.data import Dataset
import torch
from PIL import Image
import requests
from transformers import SamProcessor
import tqdm


class SamInferDataset(Dataset):

    def __init__(self, config):
        processor = SamProcessor.from_pretrained(config.data_dir + "/" +
                                                 config.weights)

        img_url = "https://hf-mirror.com/ybelkada/segment-anything/resolve/main/assets/car.png"
        raw_image = Image.open(requests.get(img_url,
                                            stream=True).raw).convert("RGB")
        input_points = [[[450, 600]]]

        inputs = processor(raw_image,
                           input_points=input_points,
                           return_tensors="pt")
        self.img = inputs["pixel_values"][0]
        self.points = inputs["input_points"][0]
        self.osize = inputs["original_sizes"][0]
        self.dsize = inputs["reshaped_input_sizes"][0]
        self.length = config.datasize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.img.clone().float(), self.points.clone().float(
        ), self.osize.clone(), self.dsize.clone()


def build_dataset(config):
    dataset = SamInferDataset(config)
    return dataset


def build_dataloader(config):
    dataset = build_dataset(config)
    loader = dl(dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=config.num_workers)

    return loader
