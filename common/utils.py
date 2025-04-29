"""
Contains functionality for utily
"""

import os
import copy
import pathlib
import random
import shutil

import pandas as pd
import matplotlib.pyplot as plt

# import wandb
import torch
import torchvision
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
from loguru import logger
from pytorch_adapt.layers import MultipleModels
from pytorch_adapt.models import Discriminator
from pytorch_adapt.containers import Models
from pytorch_adapt.utils import common_functions as c_f

from common.networks import ARCHITECTURES, Classifier


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class AUCTracker:
    def __init__(self, alpha=0.8):
        self.auc_old = None
        self.alpha = alpha

    def update(self, auc):
        if not self.auc_old:
            self.auc_old = auc
            return self.auc_old
        else:
            calc_auc = self.auc_old * self.alpha + (1 - self.alpha) * auc
            self.auc_old = auc
            return calc_auc


class EarlyStopping:
    def __init__(self, mode="min", min_delta=0.001, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if np.isnan(metrics):
            return True
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class ModelCheckpoint:
    def __init__(self, name, mode="max", path_save="models"):
        self.path_save = os.path.join(os.getcwd(), path_save)
        os.makedirs(self.path_save, exist_ok=True)  # Garante que o diretório existe

        self.mode = mode
        self.model_name = name
        self.model_save_path = os.path.join(self.path_save, self.model_name)

        # Inicialização correta do best_score
        if mode == "max":
            self.best_score = float("-inf")
        elif mode == "min":
            self.best_score = float("inf")
        else:
            raise ValueError("mode should be either 'max' or 'min'")

        self.best_epoch = None

    def __call__(self, models, score, epoch, optimizer=None, misc=None, args=None):
        if epoch is None:
            raise ValueError("Epoch cannot be None")

        update_checkpoint = False
        if self.mode == "max" and score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            update_checkpoint = True
        elif self.mode == "min" and score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            update_checkpoint = True

        if update_checkpoint:
            logger.info(
                f"[ModelCheckpoint] Saving model to: {self.model_save_path} with score {self.best_score}"
            )

            checkpoint = {
                "args": vars(args) if args is not None else None,
                "epoch": epoch,
                "models": (
                    {m: models[m].state_dict() for m in models}
                    if isinstance(models, dict)
                    else models.state_dict()
                ),
                "optimizers": (
                    [o.state_dict() for o in optimizer]
                    if optimizer is not None
                    else None
                ),
                "misc": (
                    {m: misc[m].state_dict() for m in misc}
                    if misc is not None
                    else None
                ),
            }

            torch.save(checkpoint, self.model_save_path)

    def get_best_score(self):
        return self.best_score

    def get_best_epoch(self):
        return self.best_epoch

    def load_best_model(self, args):
        # Get the G
        load_checkpoint = torch.load(self.model_save_path)
        G = ARCHITECTURES[args.G_arch]["model"](
            weights=ARCHITECTURES[args.G_arch]["weights"]
        )
        if args.G_arch in ["vgg16", "densenet161", "densenet201"]:
            G.classifier = nn.Identity()
        else:
            G.fc = nn.Identity()
        feature_dim = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "vgg16": 25088,
        }[args.G_arch]

        G_state_dict = load_checkpoint["models"]["G"]
        G.load_state_dict(G_state_dict, strict=True)
        G = G.to(args.device)

        hidden_size = {
            "resnet18": 256,
            "resnet34": 256,
            "resnet50": 512,
            "resnet101": 2048,
        }[args.G_arch]

        # Get the C
        C = Classifier(
            in_size=feature_dim,
            hidden_size=hidden_size,
            dropout=args.dropout,
            num_classes=args.num_classes,
        )
        C_state_dict = load_checkpoint["models"]["C"]
        C.load_state_dict(C_state_dict, strict=True)
        if args.algorithm == "mcd":
            C = MultipleModels(C, c_f.reinit(copy.deepcopy(C)))
        C = C.to(args.device)

        # Get the D
        D = Discriminator(in_size=feature_dim)
        D = D.to(args.device)

        return (
            Models({"G": G, "C": C, "D": D}),
            self.best_score,
            load_checkpoint["epoch"],
        )


def set_random_seed(seed=47):
    # seed setting
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directory(type_path: str, args=None):
    if type_path == "checkpoints":
        directory = os.path.join(os.getcwd(), "checkpoints")
    elif type_path == "models":
        directory = os.path.join(os.getcwd(), "models")
    elif type_path == "results":
        directory = os.path.join(os.getcwd(), "results")
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created successfully! - {directory}")
