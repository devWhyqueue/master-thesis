from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PatchFocalLoss(nn.Module):
    """Multiclass focal loss for patch classifiers."""

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa
        """Compute the focal loss value."""
        log_probs = torch.log_softmax(logits, dim=1)
        pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-((1 - pt) ** self.gamma) * log_pt).mean()


def inverse_frequency_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Build normalized inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)


def effective_number_weights(
    labels: np.ndarray, n_classes: int, beta: float
) -> torch.Tensor:
    """Build normalized effective-number weights used by class-balanced losses."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    effective = (1.0 - np.power(beta, np.maximum(counts, 1.0))) / (1.0 - beta)
    weights = 1.0 / effective
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)


def gaussian_affinity(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Return Mahbub et al.'s Gaussian class affinity matrix."""
    squared = torch.cdist(embeddings, prototypes, p=2).pow(2)
    return torch.exp(-squared / sigma)


def cfal_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    class_weights: torch.Tensor,
    sigma: float,
    margin: float,
    gamma: float,
) -> torch.Tensor:
    """Compute center-focused affinity loss from Mahbub et al."""
    affinity = gaussian_affinity(embeddings, prototypes, sigma)
    own_affinity = affinity.gather(1, targets.unsqueeze(1)).squeeze(1)
    margins = margin + affinity - own_affinity.unsqueeze(1)
    margins.scatter_(1, targets.unsqueeze(1), 0.0)
    max_margin = torch.relu(margins).sum(dim=1)

    pairwise = torch.pdist(prototypes, p=2).pow(2)
    mean_distance = (
        pairwise.mean()
        if pairwise.numel()
        else torch.tensor(0.0, device=embeddings.device)
    )
    diversity = (
        (pairwise - mean_distance).pow(2).mean()
        if pairwise.numel()
        else torch.tensor(0.0, device=embeddings.device)
    )
    vanilla_affinity = max_margin + diversity
    modulation = (1.0 - own_affinity).pow(gamma)
    return (class_weights[targets] * modulation * vanilla_affinity).mean()
