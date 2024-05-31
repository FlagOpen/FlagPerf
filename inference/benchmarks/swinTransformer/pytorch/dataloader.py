import torchvision as tv
from torch.utils.data import DataLoader as dl
import torch
import tqdm


def build_dataset(config):
    crop = 256
    c_crop = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if config.fp16:

        class ToFloat16(object):

            def __call__(self, tensor):
                return tensor.to(dtype=torch.float16)

        tx = tv.transforms.Compose([
            tv.transforms.Resize(crop),
            tv.transforms.CenterCrop(c_crop),
            tv.transforms.ToTensor(),
            ToFloat16(),
            tv.transforms.Normalize(mean=mean, std=std),
        ])
        dataset = tv.datasets.ImageFolder(config.data_dir, tx)
    else:
        tx = tv.transforms.Compose([
            tv.transforms.Resize(crop),
            tv.transforms.CenterCrop(c_crop),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=std),
        ])
        dataset = tv.datasets.ImageFolder(config.data_dir, tx)

    return dataset


def build_dataloader(config):
    dataset = build_dataset(config)
    loader = dl(dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=config.num_workers,
                pin_memory=True)

    return loader
