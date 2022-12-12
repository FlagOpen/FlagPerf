# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

'''Download CIFAR100 Dataset with torchvision.datasets.CIFAR100.'''
import os
import sys
from torchvision.datasets import CIFAR100

def build_cifar_dataset(train_dataset_path, eval_dataset_path):
    CIFAR100(root=train_dataset_path, train=True, download=True, transform=None)
    CIFAR100(root=eval_dataset_path, train=False, download=True, transform=None)


def _make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: ", sys.argv[0], " <data_dir>")
        sys.exit(1)
    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print("Error! ", data_dir , " is not a valid directory.")
        sys.exit(2)
    train_dataset_path = os.path.join(data_dir, "train")
    eval_dataset_path = os.path.join(data_dir, "eval")
    _make_dir(train_dataset_path)
    _make_dir(eval_dataset_path)
    build_cifar_dataset(train_dataset_path, eval_dataset_path)
