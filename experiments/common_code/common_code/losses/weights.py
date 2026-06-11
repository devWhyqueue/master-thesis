"""Class weight helpers."""

from __future__ import annotations

import numpy as np
import torch


def inverse_frequency_weights(
    labels: np.ndarray, n_classes: int, power: float = 1.0
) -> torch.Tensor:
    """Build normalized inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = np.power(1.0 / np.maximum(counts, 1.0), power)
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)


def effective_number_weights(
    labels: np.ndarray, n_classes: int, beta: float = 0.999, power: float = 1.0
) -> torch.Tensor:
    """Compute effective-number class weights."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    effective = (1.0 - np.power(beta, np.maximum(counts, 1.0))) / (1.0 - beta)
    weights = np.power(1.0 / effective, power)
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)
