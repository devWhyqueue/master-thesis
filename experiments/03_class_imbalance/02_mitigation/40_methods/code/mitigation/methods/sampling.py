"""Per-row training weights for balanced sampling and GCL's stage-two sampler."""

from __future__ import annotations

import numpy as np
import torch
from imbalance_benchmark.modeling.losses import effective_number

__all__ = ["sample_weights", "draw_indices"]


def _power_class_weight(counts: np.ndarray, param: float) -> np.ndarray:
    """Class weight n_c^-param (Eq. balanced-sampling); param=0 is uniform empirical."""
    return np.power(np.maximum(counts, 1.0), -param)


def _gcl_class_weight(
    counts: np.ndarray, delta: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Effective-number class weight r_j=(1-beta_j)/(1-beta_j**n_j) (Eq. gcl-cben)."""
    span = delta.max() - delta.min()
    beta = a + b * (delta - delta.min()) / span if span > 0 else np.full_like(delta, a)
    return 1.0 / effective_number(counts, beta)


def sample_weights(
    y: np.ndarray,
    num_classes: int,
    kind: str,
    param: float = 0.0,
    delta: np.ndarray | None = None,
    a: float | None = None,
    b: float | None = None,
) -> np.ndarray:
    """Per-row sampling weights, normalized to sum to one.

    ``kind="power"``: strength ``param`` in [0, 1], ``param=1`` is fully class-balanced.
    ``kind="gcl"``: needs ``delta`` (``gcl_delta`` of the class counts) and the effective-number
    range ``a``, ``b``.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if kind == "power":
        class_weight = _power_class_weight(counts, param)
    elif kind == "gcl":
        if delta is None or a is None or b is None:
            raise ValueError("gcl sampling needs delta, a, and b")
        class_weight = _gcl_class_weight(counts, delta, a, b)
    else:
        raise ValueError(f"unknown sampling kind: {kind}")
    row_weight = class_weight[y]
    return row_weight / row_weight.sum()


def draw_indices(
    weights: np.ndarray, num_draws: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw ``num_draws`` row indices with replacement from a normalized weight vector."""
    return torch.multinomial(
        torch.from_numpy(weights), num_draws, replacement=True, generator=generator
    )
