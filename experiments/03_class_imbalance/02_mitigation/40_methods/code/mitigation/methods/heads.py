"""Cosine classifier head (GCL), DisAlign's learned logit calibration, and cRT's reinit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CosineHead", "DisAlign", "reinit_linear_head"]


class CosineHead(nn.Module):
    """Normalized-weight, normalized-feature cosine classifier; returns cos(theta), unscaled."""

    def __init__(self, dim: int, num_classes: int) -> None:
        """Initialize the cosine head's class-weight matrix."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return cos(theta) between each normalized feature and each class weight."""
        return F.linear(F.normalize(features, dim=-1), F.normalize(self.weight, dim=-1))


class DisAlign(nn.Module):
    """Stage-two logit calibration (Eq. disalign-calibration): frozen head, learned alpha/beta/v."""

    def __init__(self, dim: int, num_classes: int) -> None:
        """Initialize alpha, beta, and the confidence direction v at zero (identity calibration)."""
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(num_classes))
        self.beta = nn.Parameter(torch.zeros(num_classes))
        self.v = nn.Parameter(torch.zeros(dim))

    def forward(self, features: torch.Tensor, logits_o: torch.Tensor) -> torch.Tensor:
        """Apply the learned, input-dependent calibration to frozen-head logits."""
        confidence = torch.sigmoid(features @ self.v).unsqueeze(-1)
        return (1 + confidence * self.alpha) * logits_o + confidence * self.beta


def reinit_linear_head(head: nn.Linear) -> None:
    """Xavier-reinitialize a linear head's weight and zero its bias (cRT stage two)."""
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
