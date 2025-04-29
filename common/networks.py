"""
Contains functionality for create the backbones network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


ARCHITECTURES = {
    "resnet18": {"model": models.resnet18, "weights": models.ResNet18_Weights.DEFAULT},
    "resnet34": {"model": models.resnet34, "weights": models.ResNet34_Weights.DEFAULT},
    "resnet50": {"model": models.resnet50, "weights": models.ResNet50_Weights.DEFAULT},
    "resnet101": {
        "model": models.resnet101,
        "weights": models.ResNet101_Weights.DEFAULT,
    },
    "vgg16": {"model": models.vgg16_bn, "weights": models.VGG16_BN_Weights.DEFAULT},
    "densenet121": {
        "model": models.densenet121,
        "weights": models.DenseNet121_Weights.DEFAULT,
    },
    "densenet161": {
        "model": models.densenet161,
        "weights": models.DenseNet161_Weights.DEFAULT,
    },
    "densenet169": {
        "model": models.densenet169,
        "weights": models.DenseNet169_Weights.DEFAULT,
    },
    "densenet201": {
        "model": models.densenet201,
        "weights": models.DenseNet201_Weights.DEFAULT,
    },
    "efficientnet_b0": {
        "model": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
    },
    "efficientnet_b4": {
        "model": models.efficientnet_b4,
        "weights": models.EfficientNet_B4_Weights.DEFAULT,
    },
}


class Classifier(nn.Module):
    def __init__(self, in_size=2048, hidden_size=512, dropout=0.5, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)
