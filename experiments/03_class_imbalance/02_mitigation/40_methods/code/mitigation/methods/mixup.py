"""MixUp: convex combinations of random batch pairs and their labels (Eq. mixup-input/-loss)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["mixup_batch", "mixup_loss"]


def mixup_batch(
    x: torch.Tensor, alpha: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mix a batch with a random permutation partner; alpha<=0 leaves it unmixed (lambda=1)."""
    if alpha <= 0:
        identity = torch.arange(len(x), device=x.device)
        return x, identity, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(len(x), generator=generator, device=x.device)
    mixed = lam * x + (1 - lam) * x[perm]
    return mixed, perm, lam


def mixup_loss(
    logits: torch.Tensor, y: torch.Tensor, perm: torch.Tensor, lam: float
) -> torch.Tensor:
    """Lambda-weighted CE against the original and permuted-partner labels (Eq. mixup-loss)."""
    return lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(
        logits, y[perm]
    )
