import torch

from ..layers import EntropyLoss
from .simple_loss_validator import SimpleLossValidator


class EntropyValidator(SimpleLossValidator):
    """
    Returns the negative of the
    [entropy][pytorch_adapt.layers.entropy_loss.EntropyLoss]
    of all logits.
    """

    @property
    def loss_fn(self):
        return EntropyLoss(after_softmax=self.layer == "preds")


class EntropyCombinedValidator(EntropyValidator):
    """
    Returns the negative of the
    [entropy][pytorch_adapt.layers.entropy_loss.EntropyLoss]
    of all logits.
    """

    def __call__(self, src_train, target_train):
        combined = {
            self.layer: torch.cat([src_train[self.layer], target_train[self.layer]])
        }
        return super().__call__(target_train=combined)
