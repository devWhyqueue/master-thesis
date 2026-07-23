from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    weighted_ece,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.ordinal import (
    ordinal_metrics,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.weighted import (
    weighted_mean as _weighted_mean,
)

__all__ = ["secondary_seed_metrics"]


def _class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    class_names: list[str],
) -> dict[str, np.ndarray]:
    true_probability = np.clip(
        probabilities[np.arange(len(labels)), labels], 1e-12, 1.0
    )
    nll_values = -np.log(true_probability)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier_values = np.sum((probabilities - one_hot) ** 2, axis=1)
    metrics: dict[str, np.ndarray] = {}
    recalls, f1_by_class, nll_by_class = [], [], []
    for class_index, class_name in enumerate(class_names):
        true_class = labels == class_index
        predicted_class = predictions == class_index
        recall = _weighted_mean(
            (predictions == labels).astype(float), weights, true_class
        )
        precision = _weighted_mean(true_class.astype(float), weights, predicted_class)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                0.0,
            )
        class_nll = _weighted_mean(nll_values, weights, true_class)
        metrics.update(
            {
                f"recall:{class_name}": recall,
                f"f1:{class_name}": f1,
                f"nll:{class_name}": class_nll,
                f"brier:{class_name}": _weighted_mean(
                    brier_values, weights, true_class
                ),
            }
        )
        recalls.append(recall)
        f1_by_class.append(f1)
        nll_by_class.append(class_nll)
    metrics["balanced_accuracy"] = np.nanmean(np.stack(recalls), axis=0)
    metrics["macro_f1"] = np.nanmean(np.stack(f1_by_class), axis=0)
    metrics["macro_nll"] = np.nanmean(np.stack(nll_by_class), axis=0)
    metrics["negative_log_likelihood"] = _weighted_mean(nll_values, weights)
    metrics["brier_score"] = _weighted_mean(brier_values, weights)
    return metrics


def _group_mean(
    values: np.ndarray,
    weights: PatientWeights,
    groups: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Macro-average ``values`` over ``groups`` (e.g. slides), weighted by each
    group's (single) patient weight -- not by the group's row count.

    Folds the ``1 / bincount(codes)[codes]`` group-size correction into the
    values vector so both terms reduce to :meth:`PatientWeights.sums` calls,
    instead of allocating a full ``(n_rows, n_replicates)`` scaled weight copy.
    """
    selected = np.ones(len(values), dtype=bool) if mask is None else mask
    codes, _ = pd.factorize(groups[selected], sort=False)
    counts = np.bincount(codes)
    scale = np.zeros(len(values), dtype=np.float64)
    scale[selected] = 1.0 / counts[codes]
    numerator = weights.sums(values * scale)
    denominator = weights.sums(scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _group_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: PatientWeights,
    groups: np.ndarray,
) -> np.ndarray:
    correct = (predictions == labels).astype(float)
    recalls = []
    for label in np.unique(labels):
        selected = labels == label
        recalls.append(_group_mean(correct, weights, groups, mask=selected))
    return np.nanmean(np.stack(recalls), axis=0)


def _group_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: PatientWeights,
    groups: np.ndarray,
) -> np.ndarray:
    codes, _ = pd.factorize(groups, sort=False)
    scores = np.asarray(
        [
            f1_score(
                labels[codes == code],
                predictions[codes == code],
                average="macro",
                zero_division=0,  # type: ignore
            )
            for code in range(codes.max() + 1)
        ]
    )
    first_rows = np.unique(codes, return_index=True)[1]
    values = np.zeros(len(labels), dtype=np.float64)
    values[first_rows] = scores
    mask = np.zeros(len(labels), dtype=bool)
    mask[first_rows] = True
    return _weighted_mean(values, weights, mask)


def _cluster_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    slide_ids: np.ndarray,
    case_ids: np.ndarray,
    is_mil: bool,
) -> dict[str, np.ndarray]:
    correct = (predictions == labels).astype(float)
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    result = (
        {} if is_mil else {"patch_micro_accuracy": _weighted_mean(correct, weights)}
    )
    for name, groups in (("slide", slide_ids), ("patient", case_ids)):
        result.update(
            {
                f"{name}_macro_accuracy": _group_mean(correct, weights, groups),
                f"{name}_macro_balanced_accuracy": _group_balanced_accuracy(
                    labels, predictions, weights, groups
                ),
                f"{name}_macro_f1": _group_macro_f1(
                    labels, predictions, weights, groups
                ),
                f"{name}_macro_nll": _group_mean(nll, weights, groups),
                f"{name}_macro_brier": _group_mean(brier, weights, groups),
            }
        )
    return result


def _tier_metrics(
    metrics: dict[str, np.ndarray],
    class_names: list[str],
    tiers: dict[str, str],
) -> dict[str, np.ndarray]:
    result = {}
    for tier in ("head", "body", "tail"):
        members = [name for name in class_names if tiers.get(name) == tier]
        for metric in ("recall", "nll", "brier"):
            values = [metrics[f"{metric}:{name}"] for name in members]
            if values:
                result[f"tier_{metric}:{tier}"] = np.nanmean(np.stack(values), axis=0)
    return result


def secondary_seed_metrics(
    sample: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: PatientWeights,
    class_names: list[str],
    tiers: dict[str, str],
    identity: tuple[np.ndarray, np.ndarray],
    *,
    is_mil: bool = False,
    ordinal: bool = False,
) -> dict[str, np.ndarray]:
    """Compute the full secondary endpoint set for one model seed.

    ``sample`` is ``(labels, predictions, probabilities)`` and ``identity`` is
    ``(slide_ids, case_ids)`` for that seed's rows.
    """
    labels, predictions, probabilities = sample
    slide_ids, case_ids = identity
    metrics = _class_metrics(labels, predictions, probabilities, weights, class_names)
    metrics.update(_tier_metrics(metrics, class_names, tiers))
    metrics.update(
        {
            "accuracy": _weighted_mean((predictions == labels).astype(float), weights),
            "expected_calibration_error": weighted_ece(labels, probabilities, weights),
        }
    )
    if ordinal:
        metrics.update(ordinal_metrics(labels, predictions, weights, len(class_names)))
    metrics.update(
        _cluster_metrics(
            labels, predictions, probabilities, weights, slide_ids, case_ids, is_mil
        )
    )
    return metrics
