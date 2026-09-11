"""Intraclass correlation (ICC) estimation and effective support computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.predictors.signals.icc import icc_estimate

__all__ = [
    "ICC_CASE_CAP",
    "ICC_PATCH_CAP",
    "design_effect",
    "effective_support",
    "cell_effective_support",
    "pca_leading_direction",
    "sample_class_scores",
    "compute_class_icc",
]

ICC_CASE_CAP = 200
ICC_PATCH_CAP = 200


def design_effect(m: int, icc: float) -> float:
    """Eq. (81): DE_c(m) = 1 + (m - 1) * ICC_c."""
    return 1.0 + (float(m) - 1.0) * float(icc)


def effective_support(g: int, m: int, icc: float) -> float:
    """Eq. (81): N_eff,c(G, m) = G * m / [1 + (m - 1) * ICC_c]."""
    de = design_effect(m, icc)
    return (float(g) * float(m)) / de if de > 0 else 0.0


def cell_effective_support(
    g: int, m: int, class_iccs: dict[str, float] | list[float]
) -> float:
    """Mean redundancy-adjusted effective support averaged over all classes."""
    values = list(class_iccs.values()) if isinstance(class_iccs, dict) else class_iccs
    if not values:
        return float(g * m)
    effs = [effective_support(g, m, icc) for icc in values]
    return float(np.mean(effs))


def pca_leading_direction(features: np.ndarray) -> np.ndarray:
    """Leading principal direction of centered features via SVD."""
    centered = features - np.mean(features, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def sample_class_scores(
    class_features: np.ndarray,
    case_ids: np.ndarray,
    direction: np.ndarray,
    rng: np.random.Generator,
    case_cap: int = ICC_CASE_CAP,
    patch_cap: int = ICC_PATCH_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Capped sample of scalar projected scores and case IDs for one class."""
    if len(case_ids) == 0:
        return np.array([]), np.array([])

    unique_cases = np.unique(case_ids)
    n_cases_to_take = min(case_cap, len(unique_cases))
    chosen_cases = rng.choice(unique_cases, size=n_cases_to_take, replace=False)

    chosen_indices: list[int] = []
    chosen_case_list: list[str] = []

    for case in chosen_cases:
        case_mask = np.flatnonzero(case_ids == case)
        n_patches_to_take = min(patch_cap, len(case_mask))
        take = rng.choice(case_mask, size=n_patches_to_take, replace=False)
        chosen_indices.extend(take)
        chosen_case_list.extend([str(case)] * len(take))

    sampled_features = class_features[chosen_indices]
    scores = sampled_features @ direction
    return scores, np.asarray(chosen_case_list)


def compute_class_icc(
    class_features: np.ndarray,
    case_ids: np.ndarray,
    direction: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute one-way ANOVA ICC with unequal cluster correction on projected scores."""
    scores, sampled_cases = sample_class_scores(
        class_features, case_ids, direction, rng
    )
    est = icc_estimate(scores, sampled_cases)
    return float(est) if est is not None else 0.0

