# src/losses/reconstruction.py

import torch.nn.functional as F


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def l1_loss(pred, target):
    return F.l1_loss(pred, target)
