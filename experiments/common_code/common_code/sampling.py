"""Balanced sampling utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


BALANCED_SAMPLER_PATCH_METHODS = frozenset(
    {
        "patch_balanced_sampler_ce",
        "patch_ce_soft_f1_balanced",
        "patch_ce_soft_mcc_balanced",
        "patch_feature_balanced_sampler_ce",
        "patch_feature_ce_soft_f1_balanced",
        "patch_feature_ce_soft_mcc_balanced",
    }
)

BALANCED_SAMPLER_WSI_METHODS = frozenset(
    {"mil_balanced_sampler_ce", "sc_mil", "rankmix_mil"}
)


def uses_balanced_sampler(method: str) -> bool:
    """Return whether training should use class-balanced oversampling."""
    return method in BALANCED_SAMPLER_PATCH_METHODS


def weighted_random_sampler(
    labels: np.ndarray, generator: torch.Generator, sampler_power: float = 1.0
) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    sample_weights = [
        float((1.0 / max(counts[label], 1)) ** sampler_power) for label in labels
    ]
    return WeightedRandomSampler(sample_weights, len(sample_weights), True, generator)


class BatchBalancingSampler(Sampler[int]):
    """Sample class indices with inverse-frequency power weighting."""

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        sampler_power: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.labels = labels
        self.batch_size = batch_size
        self.sampler_power = sampler_power
        self.generator = torch.Generator().manual_seed(seed)
        self.class_indices = {
            class_idx: np.where(labels == class_idx)[0]
            for class_idx in np.unique(labels)
        }
        counts = np.bincount(labels)
        weights = np.power(1.0 / np.maximum(counts, 1), sampler_power)
        self.class_probs = weights / weights.sum()

    def __iter__(self):
        for _ in range(len(self)):
            class_idx = int(
                np.random.choice(len(self.class_probs), p=self.class_probs)
            )
            indices = self.class_indices[class_idx]
            yield int(indices[torch.randint(len(indices), (1,)).item()])

    def __len__(self) -> int:
        return len(self.labels)
