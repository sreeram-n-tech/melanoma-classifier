"""
model/losses.py -- focal loss, for the focal-loss-vs-class-weighted-CE
comparison experiment.
"""
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Standard multi-class focal loss (Lin et al., 2017). gamma=0 reduces to
    plain (optionally class-weighted) cross-entropy.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = (-ce_loss).exp()
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
