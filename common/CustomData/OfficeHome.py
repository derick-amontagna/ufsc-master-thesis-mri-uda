"""
Contains functionality for creating PyTorch Image Folder Custom
"""

import os
import pathlib
import torch

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

from loguru import logger
from common.utils import find_classes


class CustomOfficeHome(Dataset):
    def __init__(self, root="data\\OfficeHome", domain="Art", transform=None) -> None:
        logger.info(
            f"Loading Image and Label data from OfficeHome [{domain}]".center(70, "+")
        )
        self.data_path = os.path.join(os.getcwd(), root, domain)
        self.paths = list(pathlib.Path(self.data_path).glob("*/*.jpg"))

        logger.info(f"Loading Label Data".center(70, "+"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(self.data_path)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
