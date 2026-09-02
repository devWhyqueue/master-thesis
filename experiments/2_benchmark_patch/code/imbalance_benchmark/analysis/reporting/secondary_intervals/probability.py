from __future__ import annotations

import numpy as np

from typing import Any

from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    gather_seed_resampled,
    weighted_ece,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.weighted import (
    weighted_mean as _weighted_mean,
)

__all__ = ["probability_seed_metrics"]


def _probability_class_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    class_names: list[str],
) -> dict[str, np.ndarray]:
    n_classes = len(class_names)
    true_probability = np.clip(
        probabilities[np.arange(len(labels)), labels], 1e-12, 1.0
    )
    nll_values = -np.log(true_probability)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier_values = np.sum((probabilities - one_hot) ** 2, axis=1)

    class_counts = weights.class_sums(1.0, labels, n_classes)
    nll_sums = weights.class_sums(nll_values, labels, n_classes)
    brier_sums = weights.class_sums(brier_values, labels, n_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        nll_by_class = np.where(
            class_counts > 0, nll_sums / np.maximum(class_counts, 1e-12), np.nan
        )
        brier_by_class = np.where(
            class_counts > 0, brier_sums / np.maximum(class_counts, 1e-12), np.nan
        )

    metrics: dict[str, np.ndarray] = {}
    for class_index, class_name in enumerate(class_names):
        metrics[f"nll:{class_name}"] = nll_by_class[class_index]
        metrics[f"brier:{class_name}"] = brier_by_class[class_index]
    metrics["macro_nll"] = np.nanmean(nll_by_class, axis=0)
    metrics["negative_log_likelihood"] = _weighted_mean(nll_values, weights)
    metrics["brier_score"] = _weighted_mean(brier_values, weights)
    return metrics


def _group_mean(
    values: np.ndarray,
    weights: PatientWeights,
    codes: np.ndarray,
    n_groups: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Macro-average ``values`` over groups (e.g. slides) identified by integer
    ``codes``, weighted by each group's (single) patient weight -- not by the
    group's row count.

    Folds the ``1 / bincount(codes)[codes]`` group-size correction into the
    values vector so both terms reduce to :meth:`PatientWeights.sums` calls,
    instead of allocating a full ``(n_rows, n_replicates)`` scaled weight copy.
    """
    selected = np.ones(len(values), dtype=bool) if mask is None else mask
    counts = np.bincount(codes[selected], minlength=n_groups)
    scale = np.zeros(len(values), dtype=np.float64)
    scale[selected] = 1.0 / counts[codes[selected]]
    numerator = weights.sums(values * scale)
    denominator = weights.sums(scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _cluster_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    slide_codes: np.ndarray,
    case_codes: np.ndarray,
) -> dict[str, np.ndarray]:
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    result = {}
    for name, codes in (("slide", slide_codes), ("patient", case_codes)):
        n_groups = int(codes.max()) + 1 if codes.size else 0
        for metric, values in (("nll", nll), ("brier", brier)):
            result[f"{name}_macro_{metric}"] = _group_mean(
                values, weights, codes, n_groups
            )
    return result


def probability_seed_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    class_names: list[str],
    tiers: dict[str, str],
    identity: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute only probability-dependent secondary endpoints for one model seed."""
    metrics = _probability_class_metrics(labels, probabilities, weights, class_names)
    metrics["expected_calibration_error"] = weighted_ece(labels, probabilities, weights)
    for tier in ("head", "body", "tail"):
        members = [name for name in class_names if tiers.get(name) == tier]
        if members:
            for metric in ("nll", "brier"):
                metrics[f"tier_{metric}:{tier}"] = np.nanmean(
                    np.stack([metrics[f"{metric}:{name}"] for name in members]), axis=0
                )
    metrics.update(_cluster_metrics(labels, probabilities, weights, *identity))
    return metrics


def _secondary_distributions(
    context: Any,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    tiers: dict[str, str],
) -> dict[str, np.ndarray]:
    """Return paired-seed distributions for probability-only endpoints."""
    per_seed = [
        probability_seed_metrics(
            labels,
            probabilities[index],
            context.weights,
            class_names,
            tiers,
            (context.slide_codes, context.case_codes),
        )
        for index in range(probabilities.shape[0])
    ]
    seed_indices = context._paired_seed_indices(probabilities.shape[0])
    return {
        endpoint: gather_seed_resampled(
            np.stack([seed_metrics[endpoint] for seed_metrics in per_seed]),
            seed_indices,
        )
        for endpoint in per_seed[0]
    }
