import os
import pathlib
import torch
import numpy as np
import nibabel as nib
from torch.utils.data.dataset import Dataset
from skimage.transform import resize
from typing import Tuple

from common.utils import find_classes


def customToTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic)
        img = torch.unsqueeze(img, 0)
        return img.float()


class Dataset3D(Dataset):
    def __init__(
        self,
        domain=None,
        split=None,
        transform=None,
    ) -> None:
        self.data_path = os.path.join(domain, split)
        self.paths = list(pathlib.Path(self.data_path).glob("*/*.nii.gz"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(self.data_path)

    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = nib.load(image_path)
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        img = np.array(img.get_fdata()[:, :, :]).squeeze().astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = resize(img, (224, 218, 224), mode="constant")

        return customToTensor(img), class_idx

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
