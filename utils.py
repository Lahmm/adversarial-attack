import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device(use_cuda: bool = True) -> torch.device:
    # 优先使用 CUDA，不可用时回退到 CPU
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset(
    name: str,
    batch_size: int = 16,
    train: bool = False,
    num_workers: int = 2,
    data_dir: str = "dataset",
) -> Tuple[DataLoader, int]:

    name = name.strip().lower()
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
        num_classes = 10
    elif name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
        num_classes = 10
    else:
        raise ValueError(f"不支持的数据集名称: {name}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
    )
    return loader, num_classes
