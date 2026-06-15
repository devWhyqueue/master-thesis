"""Multiclass focal loss."""

from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        gamma: float,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean focal loss over the batch."""
        log_probs = torch.log_softmax(logits, dim=1)
        pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt).clamp(min=1e-6) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


PatchFocalLoss = FocalLoss
