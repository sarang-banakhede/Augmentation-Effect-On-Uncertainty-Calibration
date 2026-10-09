import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).flatten(1)
        flat_targets = targets.flatten(1)
        intersection = (probs * flat_targets).sum(1)
        dice = (2 * intersection + self.smooth) / (probs.sum(1) + flat_targets.sum(1) + self.smooth)
        return self.bce(logits, targets) + (1 - dice.mean())
