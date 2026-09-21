"""ImageNet-1K input pipeline for the CV example."""

from __future__ import annotations

import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from .config import CVConfig


class HFImageNetDataset(torch.utils.data.Dataset):
    """Read one ImageNet split from a HuggingFace datasets Arrow cache."""

    def __init__(self, cache_dir: str, *, train: bool, transform=None):
        import datasets as hf_datasets

        self.dataset = hf_datasets.load_dataset(
            "imagenet-1k",
            cache_dir=cache_dir,
            split="train" if train else "validation",
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample["label"]


def build_transform(config: CVConfig, *, train: bool):
    if train:
        return create_transform(
            input_size=config.image_size,
            is_training=True,
            color_jitter=config.color_jitter,
            auto_augment=config.auto_augment,
            interpolation="bicubic",
            re_prob=config.random_erasing_probability,
            re_mode="pixel",
            re_count=1,
        )

    resize_size = int((256 / 224) * config.image_size)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def build_imagenet(cache_dir: str, config: CVConfig):
    train_dataset = HFImageNetDataset(
        cache_dir,
        train=True,
        transform=build_transform(config, train=True),
    )
    validation_dataset = HFImageNetDataset(
        cache_dir,
        train=False,
        transform=build_transform(config, train=False),
    )
    return train_dataset, validation_dataset
