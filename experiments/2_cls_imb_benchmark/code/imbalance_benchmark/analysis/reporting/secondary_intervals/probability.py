from __future__ import annotations

import numpy as np

from typing import Any

from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    gather_seed_resampled,
    weighted_ece,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.metrics import (
    _group_mean,
    _probability_class_metrics,
)

__all__ = ["probability_seed_metrics"]


def _cluster_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    slide_ids: np.ndarray,
    case_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    return {
        f"{name}_macro_{metric}": _group_mean(values, weights, groups)
        for name, groups in (("slide", slide_ids), ("patient", case_ids))
        for metric, values in (("nll", nll), ("brier", brier))
    }


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
            (context.slide_ids, context.case_ids),
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
