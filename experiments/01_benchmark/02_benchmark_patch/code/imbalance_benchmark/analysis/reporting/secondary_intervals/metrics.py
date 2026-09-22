from __future__ import annotations

import numpy as np

from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    weighted_ece,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.ordinal import (
    ordinal_metrics,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.probability import (
    _group_mean,
    _probability_class_metrics,
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
    n_classes = len(class_names)
    metrics = _probability_class_metrics(labels, probabilities, weights, class_names)
    correct = (predictions == labels).astype(np.float64)

    # recall_c = (true positives of c) / (rows labelled c); precision_c = (true
    # positives of c) / (rows predicted c) -- both reduce to one class_sums of
    # `correct` grouped by the respective code, since `correct` is 1 only on
    # true positives (see plan §2 for the derivation).
    label_counts = weights.class_sums(1.0, labels, n_classes)
    prediction_counts = weights.class_sums(1.0, predictions, n_classes)
    true_positives_by_label = weights.class_sums(correct, labels, n_classes)
    true_positives_by_prediction = weights.class_sums(correct, predictions, n_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_by_class = np.where(
            label_counts > 0,
            true_positives_by_label / np.maximum(label_counts, 1e-12),
            np.nan,
        )
        precision_by_class = np.where(
            prediction_counts > 0,
            true_positives_by_prediction / np.maximum(prediction_counts, 1e-12),
            np.nan,
        )
        f1_by_class = np.where(
            precision_by_class + recall_by_class > 0,
            2
            * precision_by_class
            * recall_by_class
            / (precision_by_class + recall_by_class),
            0.0,
        )

    for class_index, class_name in enumerate(class_names):
        metrics[f"recall:{class_name}"] = recall_by_class[class_index]
        metrics[f"f1:{class_name}"] = f1_by_class[class_index]
    metrics["balanced_accuracy"] = np.nanmean(recall_by_class, axis=0)
    metrics["macro_f1"] = np.nanmean(f1_by_class, axis=0)
    return metrics


def _group_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: PatientWeights,
    codes: np.ndarray,
    n_groups: int,
    n_classes: int,
) -> np.ndarray:
    """Macro recall over classes, each averaged group-first (one vote per group)."""
    correct = (predictions == labels).astype(np.float64)
    group_class_counts = np.bincount(
        codes * n_classes + labels, minlength=n_groups * n_classes
    ).reshape(n_groups, n_classes)
    # Each row's group-within-its-own-class size; a divide-by-zero here is
    # impossible since every row is itself a member counted in its own cell.
    scale = 1.0 / group_class_counts[codes, labels]
    numerator = weights.class_sums(correct * scale, labels, n_classes)
    denominator = weights.class_sums(scale, labels, n_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_by_class = np.where(
            denominator > 0, numerator / np.maximum(denominator, 1e-12), np.nan
        )
    return np.nanmean(recall_by_class, axis=0)


def _group_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: PatientWeights,
    codes: np.ndarray,
    n_groups: int,
    n_classes: int,
) -> np.ndarray:
    """Per-group macro F1 (sklearn ``average="macro", zero_division=0`` semantics),
    macro-averaged over only the classes present (in truth or prediction) in that
    group, then weighted-averaged over groups by each group's one representative row.

    ``f1 = 2*tp / (support + predicted_count)`` is the standard identity
    ``2TP / (2TP + FP + FN)``, algebraically equal to ``2*precision*recall /
    (precision + recall)`` whenever that ratio is defined, and matching
    sklearn's ``zero_division=0`` fallback in every corner case (see plan §4
    and ``test_group_macro_f1_matches_sklearn_per_group``).
    """
    correct = (labels == predictions).astype(np.float64)
    support = np.bincount(
        codes * n_classes + labels, minlength=n_groups * n_classes
    ).reshape(n_groups, n_classes)
    predicted_count = np.bincount(
        codes * n_classes + predictions, minlength=n_groups * n_classes
    ).reshape(n_groups, n_classes)
    true_positive = np.bincount(
        codes * n_classes + labels, weights=correct, minlength=n_groups * n_classes
    ).reshape(n_groups, n_classes)
    label_union_size = support + predicted_count
    present = label_union_size > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_per_class = np.where(
            present, 2 * true_positive / np.maximum(label_union_size, 1), 0.0
        )
    present_count = present.sum(axis=1)
    macro_f1_by_group = np.where(
        present_count > 0,
        f1_per_class.sum(axis=1) / np.maximum(present_count, 1),
        np.nan,
    )

    first_rows = np.unique(codes, return_index=True)[1]
    values = np.zeros(len(labels), dtype=np.float64)
    values[first_rows] = macro_f1_by_group[codes[first_rows]]
    mask = np.zeros(len(labels), dtype=bool)
    mask[first_rows] = True
    return _weighted_mean(values, weights, mask)


def _cluster_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    slide_codes: np.ndarray,
    case_codes: np.ndarray,
    is_mil: bool,
) -> dict[str, np.ndarray]:
    n_classes = probabilities.shape[1]
    correct = (predictions == labels).astype(float)
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    result = (
        {} if is_mil else {"patch_micro_accuracy": _weighted_mean(correct, weights)}
    )
    for name, codes in (("slide", slide_codes), ("patient", case_codes)):
        n_groups = int(codes.max()) + 1 if codes.size else 0
        result.update(
            {
                f"{name}_macro_accuracy": _group_mean(
                    correct, weights, codes, n_groups
                ),
                f"{name}_macro_balanced_accuracy": _group_balanced_accuracy(
                    labels, predictions, weights, codes, n_groups, n_classes
                ),
                f"{name}_macro_f1": _group_macro_f1(
                    labels, predictions, weights, codes, n_groups, n_classes
                ),
                f"{name}_macro_nll": _group_mean(nll, weights, codes, n_groups),
                f"{name}_macro_brier": _group_mean(brier, weights, codes, n_groups),
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
    ``(slide_codes, case_codes)`` -- integer group codes, factorized once per
    :class:`~imbalance_benchmark.analysis.inference.context.BootstrapContext`,
    for that seed's rows.
    """
    labels, predictions, probabilities = sample
    slide_codes, case_codes = identity
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
            labels, predictions, probabilities, weights, slide_codes, case_codes, is_mil
        )
    )
    return metrics
