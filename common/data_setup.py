"""
Contains functionality for creating PyTorch DataLoaders
"""

import os
import random

import torch
from PIL import Image
from loguru import logger
from torchvision.transforms import v2 as transforms
from src.pytorch_adapt.datasets import (
    DataloaderCreator,
    SourceDataset,
    TargetDataset,
    CombinedSourceAndTargetDataset,
    ConcatDataset,
)

from common.CustomData.MRI_NII_2D import Dataset2D

NUM_WORKERS = os.cpu_count()


class GrayscaleToRGB:
    def __call__(self, x):
        if x.size(0) == 3:
            return x
        elif x.size(0) == 1:
            return torch.cat([x, x, x], dim=0)
        else:
            raise Exception("Image is not grayscale (or even RGB).")


class RandomRotFlip:
    def __call__(self, image):
        k = torch.randint(
            0, 4, (1,)
        ).item()  # Rotação aleatória (0, 90, 180, 270 graus)
        image = torch.rot90(image, k, dims=(-2, -1))

        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-1])  # Flip horizontal
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-2])  # Flip vertical

        return image


class RandomRotate:
    def __init__(self, angle_range=(-20, 20)):
        self.angle_range = angle_range

    def __call__(self, image):
        angle = random.randint(
            self.angle_range[0], self.angle_range[1]
        )  # Ângulo aleatório
        return transforms.functional.rotate(
            image, angle, interpolation=transforms.InterpolationMode.NEAREST
        )


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            GrayscaleToRGB(),
            RandomRotFlip(),
            RandomRotate(angle_range=(-20, 20)),
            transforms.ToDtype(torch.float32),
        ]
    ),
    "val_test": transforms.Compose(
        [
            transforms.ToTensor(),
            GrayscaleToRGB(),
            transforms.ToDtype(torch.float32),
        ]
    ),
}


DOMAINS = {
    "ADNI1-GE": os.path.join(
        "data", "ADNI1-T1-AD-CN", "Image", "Preprocess", "6_step_nifti_2d", "GE"
    ),
    "ADNI1-Philips": os.path.join(
        "data", "ADNI1-T1-AD-CN", "Image", "Preprocess", "6_step_nifti_2d", "Philips"
    ),
    "ADNI1-Siemens": os.path.join(
        "data", "ADNI1-T1-AD-CN", "Image", "Preprocess", "6_step_nifti_2d", "Siemens"
    ),
    "ADNI1-GE-3D": os.path.join(
        "data", "ADNI1-T1-AD-CN", "Image", "Preprocess", "5_step_class_folders", "GE"
    ),
    "ADNI1-Philips-3D": os.path.join(
        "data",
        "ADNI1-T1-AD-CN",
        "Image",
        "Preprocess",
        "5_step_class_folders",
        "Philips",
    ),
    "ADNI1-Siemens-3D": os.path.join(
        "data",
        "ADNI1-T1-AD-CN",
        "Image",
        "Preprocess",
        "5_step_class_folders",
        "Siemens",
    ),
}


def create_dataloaders_mri_2d(
    source: str,
    target: str,
    transform_train: transforms.Compose = data_transforms["train"],
    transform_val_test: transforms.Compose = data_transforms["val_test"],
    algorithm: str = "source-only",
    validator: str = "Accuracy",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = None,
):
    logger.info(f"Loading Source and Target Datasets".center(70, "+"))
    data_output = {
        "src": {"train": {}, "val_test": {}},
        "target": {"train": {}, "val_test": {}},
    }
    for domain_side, domain in zip(["src", "target"], [source, target]):
        for split in ["train", "val", "test"]:
            data = Dataset2D(
                domain=DOMAINS[domain], split=split, transform=transform_val_test
            )
            data_output[domain_side]["val_test"][split] = data
            if split in ["train", "val"] and domain_side == "src":
                data_transform = Dataset2D(
                    domain=DOMAINS[domain], split=split, transform=transform_train
                )
                data_output[domain_side]["train"][split] = data_transform
            elif split == "train" and domain_side == "target":
                data_transform = Dataset2D(
                    domain=DOMAINS[domain], split=split, transform=transform_train
                )
                data_output[domain_side]["train"][split] = data_transform

    logger.info(f"Create Source and Target Datasets".center(70, "+"))
    dataset = {}
    if algorithm == "source-only":
        dataset["src_train"] = SourceDataset(data_output["src"]["train"]["train"])
    else:
        dataset["src_train"] = SourceDataset(data_output["src"]["val_test"]["train"])
    dataset["src_val"] = SourceDataset(data_output["src"]["val_test"]["val"])
    dataset["src_test"] = SourceDataset(data_output["src"]["val_test"]["test"])

    dataset["target_train"] = TargetDataset(data_output["target"]["val_test"]["train"])
    dataset["target_val"] = TargetDataset(data_output["target"]["val_test"]["val"])
    dataset["target_test"] = TargetDataset(data_output["target"]["val_test"]["test"])

    dataset["target_train_with_labels"] = TargetDataset(
        data_output["target"]["val_test"]["train"], domain=1, supervised=True
    )
    dataset["target_val_with_labels"] = TargetDataset(
        data_output["target"]["val_test"]["val"], domain=1, supervised=True
    )
    dataset["target_test_with_labels"] = TargetDataset(
        data_output["target"]["val_test"]["test"], domain=1, supervised=True
    )

    dataset["train"] = CombinedSourceAndTargetDataset(
        SourceDataset(data_output["src"]["train"]["train"]),
        TargetDataset(data_output["target"]["train"]["train"]),
    )

    if algorithm == "source-only":
        train_names, val_names = ["src_train"], [
            "train",
            "src_val",
            "src_test",
            "target_train",
            "target_train_with_labels",
            "target_val",
            "target_val_with_labels",
            "target_test",
            "target_test_with_labels",
        ]
    else:
        train_names, val_names = ["train"], [
            "src_train",
            "src_val",
            "src_test",
            "target_train",
            "target_train_with_labels",
            "target_val",
            "target_val_with_labels",
            "target_test",
            "target_test_with_labels",
        ]

    logger.info(f"Create Dataloader".center(70, "+"))
    dc = DataloaderCreator(
        batch_size=batch_size,
        num_workers=num_workers,
        train_names=train_names,
        val_names=val_names,
        seed=seed,
    )

    dataloaders = dc(**dataset)
    target_dataset_size = len(dataset["target_train"])
    logger.info(f"Finishing the Creation of Dataloaders".center(70, "+"))
    return dataloaders, target_dataset_size, train_names[0]
