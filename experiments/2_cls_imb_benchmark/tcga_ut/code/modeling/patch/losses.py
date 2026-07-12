"""Patch classification losses (shared implementation in common_code)."""

import numpy as np
import torch
from torch import nn

from common_code.losses.focal import PatchFocalLoss
from common_code.losses.metric import (
    ScholzCombinedLoss,
    SoftF1LossMulti,
    SoftMCCLossMulti,
)
from common_code.losses.weights import inverse_frequency_weights

__all__ = [
    "PatchFocalLoss",
    "ScholzCombinedLoss",
    "SoftF1LossMulti",
    "SoftMCCLossMulti",
    "inverse_frequency_weights",
]


def _criterion(
    method: str, labels: np.ndarray, n_classes: int, gamma: float, device: torch.device
) -> nn.Module:
    if method == "patch_weighted_ce":
        return nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(labels, n_classes).to(device)
        )
    if method == "patch_focal":
        return PatchFocalLoss(gamma)
    if method == "patch_ce_soft_f1_balanced":
        return ScholzCombinedLoss(n_classes, "f1")
    if method == "patch_ce_soft_mcc_balanced":
        return ScholzCombinedLoss(n_classes, "mcc")
    return nn.CrossEntropyLoss()
