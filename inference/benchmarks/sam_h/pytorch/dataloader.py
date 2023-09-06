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

        img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
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
